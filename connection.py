"""为 asabot 0.1.5 补充独立的 WS Token、自动重连和事件接收诊断。"""

import asyncio
import json
import random
import time

import aiohttp
from asa.core.adapter import NapCatAdapter
from asa.core.event import Event
from asa.core.logging import get_logger

logger = get_logger("connection")


class AuthError(ConnectionError):
    """WebSocket 鉴权被拒。Token 错误时重连没有意义，直接抛出并提示。"""


class NapCatConnection(NapCatAdapter):
    def __init__(
        self,
        ws_url,
        http_url,
        token,
        *,
        ws_token,
        reconnect=False,
        reconnect_initial_seconds=2.0,
        reconnect_max_seconds=60.0,
    ):
        super().__init__(ws_url, http_url, token)
        self._ws_auth_header = {"Authorization": f"Bearer {ws_token}"}
        self.reconnect = reconnect
        self.reconnect_initial_seconds = reconnect_initial_seconds
        self.reconnect_max_seconds = reconnect_max_seconds
        # 供群内状态查询使用的连接指标。
        self.stats = {"connected": False, "reconnects": 0, "connected_at": None, "last_event_at": None}

    async def run_ws_loop(self, on_event):
        """消费事件。reconnect=True 时断线自动重连（指数退避 + 抖动）。"""
        if not self.reconnect:
            return await self._run_ws_once(on_event)
        attempt = 0
        while True:
            try:
                await self._run_ws_once(on_event)
                logger.warning("WebSocket 连接已结束，准备重连")
            except AuthError:
                # Token 配置错误无法通过重连恢复，交给上层退出并提示。
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("WebSocket 连接中断：{}；将自动重连", exc)
            attempt += 1
            delay = min(self.reconnect_initial_seconds * (2 ** (attempt - 1)), self.reconnect_max_seconds)
            delay += random.uniform(0, min(1.0, delay * 0.1))
            self.stats["reconnects"] += 1
            logger.info("第 {} 次重连，{:.1f} 秒后重试（NapCat 未启动时可稍后自动恢复）", attempt, delay)
            await asyncio.sleep(delay)

    async def _run_ws_once(self, on_event):
        session = await self._ensure_session()
        async with session.ws_connect(self.ws_url, headers=self._ws_auth_header) as ws:
            logger.info("WebSocket 握手成功，等待 OneBot 事件确认鉴权")
            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        raise ConnectionError("WebSocket 接收失败") from ws.exception()
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    try:
                        raw = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.warning("WebSocket 收到非 JSON 数据，已忽略")
                        continue
                    if not isinstance(raw, dict):
                        logger.warning("WebSocket 收到非对象数据，已忽略")
                        continue
                    if raw.get("status") == "failed":
                        raise AuthError(
                            f"NapCat WebSocket 拒绝连接：retcode={raw.get('retcode')}；"
                            "请检查 .env 的 NAPCAT_WS_TOKEN 是否与 3001 的 Token 一致"
                        )
                    if not raw.get("post_type"):
                        logger.debug("WebSocket 收到 API 响应，非事件")
                        continue
                    self.stats["last_event_at"] = time.time()
                    if raw.get("post_type") == "meta_event":
                        if raw.get("meta_event_type") == "lifecycle":
                            self.stats["connected"] = True
                            self.stats["connected_at"] = time.time()
                            logger.info("WebSocket 鉴权成功，已收到生命周期事件")
                        else:
                            logger.debug("WebSocket 元事件：{}", raw.get("meta_event_type"))
                    elif raw.get("post_type") in ("message", "message_sent"):
                        event = Event(raw)
                        logger.info(
                            "收到消息：type={} group={} user={} text={!r}",
                            event.message_type, event.group_id, event.user_id, event.text,
                        )
                        if event.user_id == event.self_id:
                            logger.info("机器人自身消息：默认不触发回复，请用另一个 QQ 测试")
                    try:
                        await on_event(raw)
                    except Exception:
                        logger.exception("处理 NapCat 事件失败")
            finally:
                self.stats["connected"] = False
            raise ConnectionError(f"WebSocket 已关闭，close_code={ws.close_code}")
