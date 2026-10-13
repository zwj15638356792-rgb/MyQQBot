import asyncio
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from aiohttp import web

from connection import NapCatConnection
from main import create_bot


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sent = []
        self.events = []
        self.connected = asyncio.Event()
        self.reply_received = asyncio.Event()
        self.finished = asyncio.Event()
        self.ws_header = None
        self.http_header = None
        self.reject = False
        self.ws = None
        self.config_dir = tempfile.TemporaryDirectory()
        self.plugins_path = Path(self.config_dir.name) / "plugins.toml"
        self.plugins_path.write_text('[plugins."bot.echo"]\nenabled = true\n', encoding="utf-8")
        app = web.Application()
        app.router.add_get("/ws", self.websocket)
        app.router.add_post("/get_login_info", self.login)
        app.router.add_post("/send_group_msg", self.send_group)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        port = self.runner.addresses[0][1]
        self.http_url = f"http://127.0.0.1:{port}"
        self.ws_url = f"ws://127.0.0.1:{port}/ws"
        self.adapter = None
        self.task = None

    async def asyncTearDown(self):
        self.finished.set()
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.adapter:
            await self.adapter.close()
        await self.runner.cleanup()
        self.config_dir.cleanup()

    async def login(self, request):
        self.http_header = request.headers.get("Authorization")
        return web.json_response({"status": "ok", "retcode": 0,
                                  "data": {"user_id": 10001, "nickname": "test"}})

    async def send_group(self, request):
        self.http_header = request.headers.get("Authorization")
        self.sent.append(await request.json())
        self.reply_received.set()
        return web.json_response({"status": "ok", "retcode": 0, "data": {"message_id": 1}})

    async def websocket(self, request):
        self.ws_header = request.headers.get("Authorization")
        self.ws = web.WebSocketResponse()
        await self.ws.prepare(request)
        if self.reject:
            await self.ws.send_json({"status": "failed", "retcode": 1403, "message": "token验证失败"})
            await self.ws.close()
            return self.ws
        await self.ws.send_json({"post_type": "meta_event", "meta_event_type": "lifecycle",
                                 "sub_type": "connect", "self_id": 10001})
        self.connected.set()
        await self.finished.wait()
        await self.ws.close()
        return self.ws

    async def start_bot(self):
        with patch.dict(os.environ, {"WS_URL": self.ws_url, "HTTP_URL": self.http_url,
                                     "NAPCAT_TOKEN": "http-test", "NAPCAT_WS_TOKEN": "ws-test"}):
            bot = create_bot(plugins_path=self.plugins_path)
        self.adapter = bot.adapter
        # 本组用例依赖“连接关闭即抛出”，单独关闭自动重连；重连行为另有用例覆盖。
        self.adapter.reconnect = False
        await bot._fetch_login_info()
        self.task = asyncio.create_task(self.adapter.run_ws_loop(bot._handle_raw_event))
        await asyncio.wait_for(self.connected.wait(), 3)
        return bot

    def message(self, text, user=12345, *, array_only=False):
        raw = {"post_type": "message", "message_type": "group", "sub_type": "normal",
               "self_id": 10001, "group_id": 67890, "user_id": user,
               "message_id": 2, "sender": {"user_id": user},
               "message": [{"type": "text", "data": {"text": text}}]}
        if not array_only:
            raw["raw_message"] = text
        return raw

    async def test_distinct_tokens_and_single_ping_reply(self):
        bot = await self.start_bot()
        self.assertEqual(len(bot.handler.get_handlers()), 1)
        await self.ws.send_json(self.message("ping"))
        await asyncio.wait_for(self.reply_received.wait(), 3)
        self.assertEqual(self.ws_header, "Bearer ws-test")
        self.assertEqual(self.http_header, "Bearer http-test")
        self.finished.set()
        with self.assertRaises(ConnectionError):
            await asyncio.wait_for(self.task, 3)
        self.assertEqual(self.sent, [{"group_id": 67890,
                                     "message": "[CQ:at,qq=12345] pong!", "auto_escape": False}])

    async def test_array_text_matching_and_ignored_messages(self):
        await self.start_bot()
        await self.ws.send_json(self.message("hello"))
        await self.ws.send_json(self.message("ping", user=10001))
        await self.ws.send_json(self.message("hello ping", array_only=True))
        await asyncio.wait_for(self.reply_received.wait(), 3)
        self.finished.set()
        with self.assertRaises(ConnectionError):
            await asyncio.wait_for(self.task, 3)
        self.assertEqual(len(self.sent), 1)

    async def test_auth_rejection_is_reported_and_never_dispatched(self):
        self.reject = True
        self.adapter = NapCatConnection(self.ws_url, self.http_url, "http-test", ws_token="bad")

        async def collect(raw):
            self.events.append(raw)

        with self.assertRaisesRegex(ConnectionError, "1403.*NAPCAT_WS_TOKEN"):
            await asyncio.wait_for(self.adapter.run_ws_loop(collect), 3)
        self.assertEqual(self.events, [])

    async def test_socket_closure_is_reported(self):
        self.adapter = NapCatConnection(self.ws_url, self.http_url, "http-test", ws_token="ws-test")

        async def collect(raw):
            self.events.append(raw)

        self.task = asyncio.create_task(self.adapter.run_ws_loop(collect))
        await asyncio.wait_for(self.connected.wait(), 3)
        self.finished.set()
        with self.assertRaisesRegex(ConnectionError, "WebSocket 已关闭"):
            await asyncio.wait_for(self.task, 3)
        self.assertEqual(self.events[0]["meta_event_type"], "lifecycle")


class ReconnectTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.connections = 0
        self.events = []
        self.delivered = asyncio.Event()
        self.message_received = asyncio.Event()
        self.mode = "drop_first"
        app = web.Application()
        app.router.add_get("/ws", self.websocket)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = self.runner.addresses[0][1]
        self.adapter = NapCatConnection(
            f"ws://127.0.0.1:{port}/ws", f"http://127.0.0.1:{port}", "t", ws_token="t",
            reconnect=True, reconnect_initial_seconds=0.05, reconnect_max_seconds=0.1,
        )
        self.task = None

    async def asyncTearDown(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.adapter.close()
        await self.runner.cleanup()

    async def websocket(self, request):
        self.connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        if self.mode == "reject":
            await ws.send_json({"status": "failed", "retcode": 1403, "message": "token验证失败"})
            await ws.close()
            return ws
        await ws.send_json({"post_type": "meta_event", "meta_event_type": "lifecycle", "self_id": 1})
        if self.connections == 1:
            await ws.close()  # 模拟第一次连接被断开
            return ws
        await ws.send_json({"post_type": "message", "message_type": "group", "self_id": 1,
                            "group_id": 1, "user_id": 2,
                            "message": [{"type": "text", "data": {"text": "hi"}}]})
        self.delivered.set()
        await ws.close()
        return ws

    async def collect(self, raw):
        self.events.append(raw)
        if raw.get("post_type") == "message":
            self.message_received.set()

    async def test_disconnect_triggers_reconnect_and_delivers_events(self):
        self.task = asyncio.create_task(self.adapter.run_ws_loop(self.collect))
        await asyncio.wait_for(self.message_received.wait(), 5)
        self.assertGreaterEqual(self.connections, 2)
        self.assertGreaterEqual(self.adapter.stats["reconnects"], 1)
        self.assertTrue(any(e.get("post_type") == "message" for e in self.events))

    async def test_auth_failure_is_not_retried(self):
        self.mode = "reject"
        with self.assertRaisesRegex(ConnectionError, "1403.*NAPCAT_WS_TOKEN"):
            await asyncio.wait_for(self.adapter.run_ws_loop(self.collect), 3)
        self.assertEqual(self.connections, 1)
        self.assertEqual(self.adapter.stats["reconnects"], 0)
