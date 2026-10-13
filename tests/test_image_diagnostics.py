"""生图 HTTP 链路回归：仅连接本地模拟服务，不调用付费接口。"""

import asyncio
import base64
import unittest
from unittest.mock import patch

from aiohttp import web

from image_service import FALLBACK_CHARACTER, ImageClient, ImageError, ImageSettings


class ImageDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.status = 200
        self.delay = 0
        self.download_delay = 0
        self.requests = []
        self.entered = asyncio.Event()
        self.content = b"test-image-bytes"
        self.data = {"data": [{"b64_json": base64.b64encode(self.content).decode()}]}
        app = web.Application()
        app.router.add_post("/api/v3/images/generations", self.generate)
        app.router.add_get("/image.png", self.download)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", 0).start()
        self.root = f"http://127.0.0.1:{self.runner.addresses[0][1]}"
        settings = ImageSettings(enabled=True, api_key="diagnostic-secret",
                                 base_url=self.root + "/api/v3", model="test-endpoint",
                                 cooldown_seconds=0, output_dir=".tmp/image_diagnostics")
        self.client = ImageClient({**FALLBACK_CHARACTER, "ref_dir": ".tmp/no-image-refs"}, settings)
        self.paths = []

    async def asyncTearDown(self):
        await self.client.close()
        await self.runner.cleanup()
        for path in self.paths:
            path.unlink(missing_ok=True)

    async def generate(self, request):
        self.requests.append((request.headers.get("Authorization"), await request.json()))
        self.entered.set()
        # 模拟当前接入点：可选参数即使设为 disabled 也会被拒绝。
        if "sequential_image_generation" in self.requests[-1][1]:
            return web.json_response({"error": {"code": "InvalidParameter",
                                      "message": "sequential_image_generation is not supported"}},
                                     status=400)
        if self.delay:
            await asyncio.sleep(self.delay)
        return web.json_response(self.data, status=self.status,
                                 headers={"X-Request-Id": "request-for-debugging"})

    async def download(self, request):
        if self.download_delay:
            await asyncio.sleep(self.download_delay)
        return web.Response(body=self.content)

    async def test_generation_saves_base64_and_url_responses(self):
        for item in ({"b64_json": base64.b64encode(self.content).decode()},
                     {"url": self.root + "/image.png"}):
            with self.subTest(item=list(item)):
                self.data = {"data": [item]}
                path = await self.client.generate("在咖啡馆", user=123)
                self.paths.append(path)
                self.assertEqual(path.read_bytes(), self.content)
        authorization, payload = self.requests[0]
        self.assertEqual(authorization, "Bearer diagnostic-secret")
        self.assertEqual(payload["model"], "test-endpoint")
        self.assertIn("在咖啡馆", payload["prompt"])
        self.assertEqual(self.client.quota_left(), 28)

    async def test_http_error_preserves_request_id_and_hides_key(self):
        self.status = 404
        self.data = {"error": {"code": "EndpointNotFound",
                               "message": "endpoint missing " + "x" * 250 + " diagnostic-secret"}}
        with self.assertRaises(ImageError) as raised:
            await self.client.generate()
        detail = str(raised.exception)
        self.assertIn("404", detail)
        self.assertIn("request-for-debugging", detail)
        self.assertIn("EndpointNotFound", detail)
        self.assertIn("[已隐藏]", detail)
        self.assertNotIn("diagnostic-secret", detail)
        self.assertEqual(self.client.quota_left(), 30)

    async def test_request_and_download_timeouts_are_distinguishable(self):
        self.client.settings.timeout_seconds = 0.2
        self.delay = 0.5
        with self.assertRaisesRegex(ImageError, "生图超时"):
            await self.client.generate()
        self.delay = 0
        self.download_delay = 0.5
        self.data = {"data": [{"url": self.root + "/image.png"}]}
        with self.assertRaisesRegex(ImageError, "下载生成的图片超时"):
            await self.client.generate()
        self.assertEqual(self.client.quota_left(), 30)

    async def test_cancellation_returns_reserved_quota_and_releases_lock(self):
        self.delay = 0.06
        task = asyncio.create_task(self.client.generate())
        await asyncio.wait_for(self.entered.wait(), timeout=2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.client.quota_left(), 30)
        self.assertFalse(self.client._lock.locked())

    async def test_reference_preparation_runs_outside_event_loop(self):
        original = self.client._build_payload

        def build(prompt, refs):
            with self.assertRaises(RuntimeError):
                asyncio.get_running_loop()
            return original(prompt, refs)

        with patch.object(self.client, "_build_payload", side_effect=build):
            path = await self.client.generate()
            self.paths.append(path)
