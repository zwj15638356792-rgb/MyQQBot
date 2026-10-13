"""一键启动回归：临时配置和本地模拟 API，不启动真实 QQ 或机器人。"""

import hashlib
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import URLError

from start_bot import Launcher, NapCatAPI, StartupError, ensure_admin


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=".tmp")
        self.root = Path(self.temp.name).resolve()
        self.napcat = self.root / "napcat install (test)"
        self.config_dir = self.napcat / "napcat/config"
        self.config_dir.mkdir(parents=True)
        for name in ("napcat.bat", "node.exe", "index.js"):
            (self.napcat / name).write_bytes(b"")
        self.python = self.root / "python.exe"
        self.python.write_bytes(b"")
        (self.root / "launcher_settings.json").write_text(json.dumps({
            "napcat_dir": str(self.napcat), "python_exe": str(self.python), "account_id": "123",
        }), encoding="utf-8")
        (self.config_dir / "webui.json").write_text(json.dumps({"port": 6099, "token": "test-webui"}),
                                                   encoding="utf-8")
        (self.config_dir / "onebot11_123.json").write_text(json.dumps({"network": {
            "httpServers": [{"enable": True, "port": 3000, "token": "test-http"}],
            "websocketServers": [{"enable": True, "port": 3001, "token": "test-ws"}],
        }}), encoding="utf-8")
        self.launcher = Launcher(self.root)
        self.launcher.env.update({"HTTP_URL": "http://127.0.0.1:3000", "WS_URL": "ws://127.0.0.1:3001",
                                  "NAPCAT_TOKEN": "test-http", "NAPCAT_WS_TOKEN": "test-ws"})

    def tearDown(self):
        self.launcher.close()
        self.temp.cleanup()

    def test_validate_rejects_mismatched_token_before_starting_anything(self):
        self.launcher.validate()
        self.launcher.env["NAPCAT_WS_TOKEN"] = "wrong"
        with self.assertRaisesRegex(StartupError, "端口和 Token"):
            self.launcher.validate()
        self.assertIsNone(self.launcher.bot)
        self.assertIsNone(self.launcher.napcat)

    def prepare_online(self, account="123"):
        self.launcher.api = Mock()
        self.launcher.api.call.side_effect = lambda route: (
            {"isLogin": True} if route.endswith("CheckLoginStatus") else {"uin": account})
        self.launcher.validate = Mock()
        self.launcher.onebot_ready = Mock(return_value=True)
        original = self.launcher.emit

        def emit(kind, value):
            original(kind, value)
            if kind == "online" and value:
                self.launcher.stop.set()

        self.launcher.emit = emit

    def test_login_starts_bot_only_after_account_and_onebot_are_ready(self):
        self.prepare_online()
        child = Mock()
        child.poll.return_value = None
        with patch.object(self.launcher, "spawn", return_value=child) as spawn, \
                patch("start_bot.existing_bot", return_value=None):
            self.launcher.run()
        self.launcher.onebot_ready.assert_called_once()
        spawn.assert_called_once_with([str(self.python), "-u", "main.py"], self.root, "bot.log")
        self.assertIs(self.launcher.bot, child)
        child.poll.return_value = 0

    def test_existing_bot_is_reused_and_never_terminated(self):
        self.prepare_online()
        with patch.object(self.launcher, "spawn") as spawn, \
                patch("start_bot.existing_bot", return_value=456):
            self.launcher.run()
        spawn.assert_not_called()
        self.assertEqual(self.launcher.reused_bot_pid, 456)
        self.launcher.close()
        self.assertIsNone(self.launcher.bot)

    def test_cold_start_waits_for_scan_before_starting_bot(self):
        self.prepare_online()
        self.launcher.api.authenticate.side_effect = URLError("not running")
        states = iter([False, True])
        qr_path = self.napcat / "napcat/cache/qrcode.png"
        qr_path.parent.mkdir(parents=True)
        qr_path.write_bytes(b"placeholder-qr-image")
        original_stamp = qr_path.stat().st_mtime_ns
        napcat, bot = Mock(), Mock()
        napcat.poll.return_value = bot.poll.return_value = None

        def call(route):
            if route.endswith("CheckLoginStatus"):
                online = next(states)
                if not online:
                    self.assertIsNone(self.launcher.bot)
                    qr_path.write_bytes(b"placeholder-qr-image")
                    os.utime(qr_path, ns=(original_stamp + 1000000000, original_stamp + 1000000000))
                # 新旧二维码文件内容相同且接口未返回二维码URL，也必须显示图片。
                return {"isLogin": online}
            return {"uin": "123"}

        self.launcher.api.call.side_effect = call
        with patch.object(self.launcher, "spawn", side_effect=[napcat, bot]) as spawn, \
                patch.object(self.launcher.stop, "wait", return_value=False), \
                patch("start_bot.existing_bot", return_value=None):
            self.launcher.run()
        self.assertEqual(spawn.call_count, 2)
        command = spawn.call_args_list[0].args[0]
        self.assertIn('/d /s /c', command)
        self.assertTrue(command.endswith(f'""{self.napcat / "napcat.bat"}""'))
        self.assertEqual(spawn.call_args_list[1].args[0], [str(self.python), "-u", "main.py"])
        events = list(self.launcher.events.queue)
        self.assertIn(("qr", qr_path), events)
        napcat.poll.return_value = bot.poll.return_value = 0

    def test_cached_qr_is_hidden_until_napcat_rewrites_it(self):
        qr_path = self.napcat / "napcat/cache/qrcode.png"
        qr_path.parent.mkdir(parents=True)
        qr_path.write_bytes(b"old-qr")
        self.assertIsNone(self.launcher.ready_qr_path())
        child = Mock()
        child.poll.return_value = 0
        with patch.object(self.launcher, "spawn", return_value=child):
            self.launcher.start_napcat()
        self.assertIsNone(self.launcher.ready_qr_path())
        qr_path.write_bytes(b"new-qr")
        self.assertEqual(self.launcher.ready_qr_path(), qr_path)

    def test_qr_timeout_even_when_login_api_continues_to_respond(self):
        self.launcher.validate = Mock()
        self.launcher.api = Mock()
        self.launcher.api.call.return_value = {"isLogin": False}
        # API 每次成功都不会掩盖二维码文件始终不生成的问题。
        with patch("start_bot.time.monotonic", side_effect=[0, 0, 0, 61, 61]), \
                patch.object(self.launcher.stop, "wait", return_value=False), \
                patch.object(self.launcher, "spawn") as spawn:
            self.launcher.run()
        spawn.assert_not_called()
        self.assertTrue(any(kind == "error" and "没有更新二维码" in value
                            for kind, value in self.launcher.events.queue))

    @unittest.skipUnless(os.name == "nt", "Windows UAC")
    def test_uac_launch_exits_parent_before_services_and_lock(self):
        with patch("start_bot.is_admin", return_value=False), \
                patch("start_bot.sys.argv", ["start_bot.py"]), \
                patch("start_bot.request_admin", return_value=42):
            self.assertFalse(ensure_admin())

    @unittest.skipUnless(os.name == "nt", "Windows UAC")
    def test_uac_cancellation_has_actionable_message(self):
        with patch("start_bot.is_admin", return_value=False), \
                patch("start_bot.sys.argv", ["start_bot.py"]), \
                patch("start_bot.request_admin", return_value=5):
            with self.assertRaisesRegex(StartupError, "启动已取消"):
                ensure_admin()

    @unittest.skipUnless(os.name == "nt", "Windows UAC")
    def test_unsuccessful_elevation_does_not_repeat_uac(self):
        with patch("start_bot.is_admin", return_value=False), \
                patch("start_bot.sys.argv", ["start_bot.py", "--elevated"]), \
                patch("start_bot.request_admin") as elevate:
            with self.assertRaisesRegex(StartupError, "以管理员身份运行"):
                ensure_admin()
        elevate.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows UAC")
    def test_admin_start_does_not_request_uac_again(self):
        with patch("start_bot.is_admin", return_value=True), \
                patch("start_bot.request_admin") as elevate:
            self.assertTrue(ensure_admin())
        elevate.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows batch entry point")
    def test_batch_entry_runs_with_spaces_and_parentheses_in_path(self):
        (self.napcat / "napcat.bat").write_bytes(b"@echo off\r\necho BAT_ENTRY_OK\r\n")
        self.launcher.start_napcat()
        self.assertEqual(self.launcher.napcat.wait(timeout=5), 0)
        self.assertIn(b"BAT_ENTRY_OK", (self.launcher.runtime / "napcat.log").read_bytes())

    def test_wrong_qq_account_does_not_start_bot(self):
        self.prepare_online(account="999")
        with patch.object(self.launcher, "spawn") as spawn:
            self.launcher.run()
        spawn.assert_not_called()
        events = list(self.launcher.events.queue)
        self.assertTrue(any(kind == "error" and "不是配置的机器人账号" in value for kind, value in events))

    def test_close_only_terminates_owned_processes(self):
        for name in ("bot", "napcat"):
            child = Mock()
            child.poll.return_value = None
            setattr(self.launcher, name, child)
        self.launcher.close()
        self.launcher.bot.terminate.assert_called_once()
        self.launcher.napcat.terminate.assert_called_once()
        self.launcher.bot.poll.return_value = 0
        self.launcher.napcat.poll.return_value = 0

    def test_webui_authentication_and_login_poll_use_local_protocol(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body, self.headers.get("Authorization")))
                data = {"Credential": "test-session"} if self.path == "/api/auth/login" else {"isLogin": True}
                payload = json.dumps({"code": 0, "data": data}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            (self.config_dir / "webui.json").write_text(json.dumps({
                "port": server.server_port, "token": "test-webui",
            }), encoding="utf-8")
            api = NapCatAPI(self.napcat)
            self.assertTrue(api.call("/QQLogin/CheckLoginStatus")["isLogin"])
            self.assertEqual(requests[0][1]["hash"], hashlib.sha256(b"test-webui.napcat").hexdigest())
            self.assertEqual(requests[1][2], "Bearer test-session")
            self.assertNotIn("test-webui", requests[0][0])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
