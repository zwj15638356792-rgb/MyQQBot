import asyncio
import base64
import json
import os
import tempfile
import unittest
from datetime import datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import aiohttp
from aiohttp import web
from asa.core.event import Event
from PIL import Image

from ai_service import (AIError, AISettings, ChatClient, GroupResponder, KnowledgeBase,
                        KnowledgeEntry, Persona, build_knowledge, describe_network_error,
                        humanize_reply, is_repetitive, message_text, mood_note, sender_name,
                        similarity, split_reply, with_system_note)
from image_service import (FALLBACK_CHARACTER, ImageClient, ImageError, ImageSettings,
                           _natural_key, build_image_segment, clean_scene, collect_references,
                           compose_prompt, encode_image)
from main import create_bot


def message(text="你好", *, group=100, user=200, mentioned=True, message_id=None):
    segments = [{"type": "text", "data": {"text": text}}]
    if mentioned:
        segments.insert(0, {"type": "at", "data": {"qq": "10001"}})
    raw = {"post_type": "message", "message_type": "group", "self_id": 10001,
           "group_id": group, "user_id": user, "message": segments}
    if message_id is not None:
        raw["message_id"] = message_id
    return Event(raw)


def private_message(text="你好", *, user=200, message_id=None, card=None):
    """私聊事件。没有 group_id，也没有 @ 段（一对一里 @ 没有意义）。"""
    raw = {"post_type": "message", "message_type": "private", "sub_type": "friend",
           "self_id": 10001, "user_id": user,
           "message": [{"type": "text", "data": {"text": text}}]}
    if message_id is not None:
        raw["message_id"] = message_id
    if card:
        raw["sender"] = {"nickname": card}
    return Event(raw)


class FakeClient:
    configured = True

    def __init__(self):
        self.calls = []
        self.gate = None
        self.reply = "你好！"
        self.error = None
        # 反重复/补一句这类"第二次请求"用的预置回复与错误
        self.rewrites = None
        self.rewrite_error = None
        # 用来验证"同一时刻只有一个请求在飞"，而不是只看总调用次数
        self.inflight = 0
        self.peak = 0

    async def complete(self, messages):
        self.calls.append(messages)
        # 带"别重复"提醒的那次请求走预置回复，方便单独控制重写结果
        retrying = any("不要重复" in str(m.get("content", "")) for m in messages)
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        try:
            if self.gate:
                await self.gate.wait()
            if retrying and self.rewrite_error:
                raise self.rewrite_error
            if self.error:
                raise self.error
            if retrying and self.rewrites is not None:
                return next(self.rewrites)
            return self.reply
        finally:
            self.inflight -= 1


async def _next_reply(replies, client, messages):
    """按顺序返回预置回复，用来验证"主回复 + 补一句"这类多次调用。"""
    client.calls.append(messages)
    return next(replies)


class _FakeImageClient:
    """假生图客户端：不碰网络，只记录被要求画什么。"""

    def __init__(self, *, error=None, configured=True):
        self.settings = ImageSettings(enabled=True, api_key="test-key", daily_limit=30)
        self.character = {"ref_dir": "character_refs"}
        self.scenes = []
        self.error = error
        self._configured = configured
        self._used_today = 0

    @property
    def configured(self):
        return self._configured

    def quota_left(self):
        return max(0, self.settings.daily_limit - self._used_today)

    def blocked_reason(self, now=None, *, user=None):
        if self._used_today >= self.settings.daily_limit:
            return f"今天的 {self.settings.daily_limit} 张额度用完了"
        return None

    async def generate(self, scene="", *, now=None, user=None):
        self.scenes.append(scene)
        if self.error:
            raise self.error
        self._used_today += 1
        return Path(f"generated/fake-{len(self.scenes)}.png")


class FakeBot:
    account_id = 10001

    def __init__(self):
        self.sent = []
        # 私聊发送单独记一份，方便断言"这条到底发到群还是发到私聊"
        self.private_sent = []
        self.result = {"status": "ok", "retcode": 0}

    async def send_group(self, segments, *, group_id):
        self.sent.append((group_id, segments))
        return self.result

    async def send_private(self, segments, *, user_id):
        self.private_sent.append((user_id, segments))
        return self.result


class ResponderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = 1000
        self.settings = AISettings(allowed_groups=[100, 101])
        # 默认关掉"补回被漏掉的消息"，让限流相关的旧用例保持"丢弃即静默"的语义；
        # 补回本身由专门的用例打开验证。
        self.settings.catch_up_missed = False
        self.client = FakeClient()
        self.bot = FakeBot()
        self.slept = []
        self.responder = GroupResponder(self.settings, self.client, lambda: self.now, self._fake_sleep)

    async def _fake_sleep(self, seconds):
        """记录拟人延迟但不真的等待，否则整套用例会被拖慢几十秒。

        同时把假时钟往前推：真实的 sleeper 睡过去，时间就是在流逝，
        补回这类"等冷却结束"的逻辑依赖这一点。
        """
        self.slept.append(seconds)
        self.now += seconds

    async def asyncTearDown(self):
        await self.responder.close()

    async def settle(self):
        """等所有后台任务跑完。循环处理"任务里又派生了新任务"的情况。"""
        for _ in range(20):
            if not self.responder.tasks:
                return
            await asyncio.gather(*list(self.responder.tasks), return_exceptions=True)
        self.fail("后台任务一直在派生新任务，没有收敛")

    async def ask(self, *args, **kwargs):
        await self.responder.handle(message(*args, **kwargs), self.bot)
        await self.settle()

    def sent_texts(self):
        return [segments[-1]["data"]["text"] for _, segments in self.bot.sent]

    def use_knowledge(self, *entries, direct=0.55, context=0.15):
        self.responder.knowledge = KnowledgeBase(list(entries), direct_threshold=direct,
                                                 context_threshold=context)

    async def test_whitelist_self_plain_text_and_empty_mention_are_silent(self):
        for event in (message(group=999), message(user=10001), message(mentioned=False), message("")):
            await self.responder.handle(event, self.bot)
        await self.settle()
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.bot.sent, [])
        self.assertEqual(self.responder.history, {})

    async def test_empty_whitelist_and_disabled_are_silent(self):
        self.settings.allowed_groups = []
        await self.ask()
        self.settings.allowed_groups = [100]
        self.settings.enabled = False
        await self.ask()
        self.assertEqual(self.client.calls, [])

    async def test_keyword_modes_prefix_and_contains(self):
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        for mode, mention, text, expected in (
            ("keyword", False, "/ai 问题", True),
            ("keyword", False, "说 /ai 问题", False),
            ("mention_or_keyword", False, "/ai 问题", True),
            ("mention_or_keyword", True, "问题", True),
            ("mention_and_keyword", True, "/ai 问题", True),
            ("mention_and_keyword", False, "/ai 问题", False),
            ("mention_and_keyword", True, "问题", False),
        ):
            with self.subTest(mode=mode, mention=mention, text=text):
                self.settings.trigger_mode = mode
                before = len(self.client.calls)
                await self.ask(text, mentioned=mention)
                self.assertEqual(len(self.client.calls) - before, int(expected))
        self.settings.trigger_mode = "keyword"
        self.settings.keyword_match = "contains"
        before = len(self.client.calls)
        await self.ask("说 /ai 问题", mentioned=False)
        self.assertEqual(len(self.client.calls), before + 1)

    async def test_group_and_user_cooldown_are_independent(self):
        await self.ask()
        self.now += 5
        await self.ask(user=201)
        self.now += 6
        await self.ask()  # 同一用户仍在30秒冷却期。
        await self.ask(user=201)
        self.now += 31
        await self.ask()
        self.assertEqual(len(self.client.calls), 3)

    async def test_busy_group_drops_requests_without_queueing(self):
        self.client.gate = asyncio.Event()
        await self.responder.handle(message(), self.bot)
        await asyncio.sleep(0)
        self.now += 100
        await self.responder.handle(message(user=201), self.bot)
        self.client.gate.set()
        await self.settle()
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(len(self.bot.sent), 1)

    async def test_global_concurrency_limit_and_other_group(self):
        self.settings.max_concurrent_requests = 1
        self.settings.catch_up_missed = False
        self.client.gate = asyncio.Event()
        await self.responder.handle(message(), self.bot)
        await self.responder.handle(message(group=101), self.bot)
        self.client.gate.set()
        await self.settle()
        # 并发上限是 1：两个群都想要模型，但同一时刻只能有一个在飞
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.peak, 1)

    async def test_history_includes_previous_answer_and_isolates_groups(self):
        await self.ask("第一个问题")
        self.now += 31
        await self.ask("第二个问题")
        second = self.client.calls[1]
        self.assertEqual([m["role"] for m in second], ["system", "user", "assistant", "user"])
        self.assertIn("第一个问题", second[1]["content"])
        await self.ask("另外一个群", group=101)
        self.assertEqual(len(self.client.calls[2]), 2)

    async def test_group_history_records_plain_messages_and_expires(self):
        self.settings.memory_mode = "group"
        self.settings.history_messages = 2
        # 关闭连发合并，这样每条都是独立一条，能验证条数与过期
        self.settings.merge_consecutive_messages = False
        await self.ask("群聊文本1", mentioned=False)
        await self.ask("群聊文本2", mentioned=False)
        await self.ask("当前问题")
        request = self.client.calls[0]
        self.assertEqual(len(request), 4)
        self.assertIn("群聊文本1", request[1]["content"])
        self.assertEqual(sum("当前问题" in m["content"] for m in request if m["role"] == "user"), 1)
        self.assertEqual(len(self.responder.history[100]), 2)
        self.now += 1801
        await self.ask("新问题")
        self.assertEqual(len(self.client.calls[1]), 2)

    async def test_duplicate_events_cannot_send_twice(self):
        await self.ask(message_id=1)
        self.now += 31
        await self.ask(message_id=1)
        self.assertEqual(len(self.client.calls), 1)

    async def test_reply_is_one_bounded_text_without_generated_mentions(self):
        self.settings.max_reply_chars = 12
        self.client.reply = "[CQ:at,qq=all]@所有人 " + "很长的回答" * 20
        await self.ask()
        group, segments = self.bot.sent[0]
        self.assertEqual(group, 100)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["type"], "text")
        self.assertEqual(len(segments[0]["data"]["text"]), 12)
        self.assertNotIn("CQ:", segments[0]["data"]["text"])

    async def test_errors_are_silent_and_still_rate_limited(self):
        self.client.error = AIError("模拟HTTP 429")
        await self.ask()
        await self.ask(user=201)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.bot.sent, [])
        self.assertNotIn("assistant", [entry[1]["role"] for entry in self.responder.history[100]])

    async def test_missing_configuration_is_silent(self):
        self.client.configured = False
        await self.ask()
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.bot.sent, [])

    async def test_close_cancels_pending_request(self):
        self.client.gate = asyncio.Event()
        await self.responder.handle(message(), self.bot)
        await asyncio.sleep(0)
        await self.responder.close()
        self.assertEqual(self.bot.sent, [])
        self.assertEqual(self.responder.busy_groups, set())

    async def test_real_time_is_injected_into_system_prompt(self):
        self.responder.time_provider = lambda: datetime(2026, 10, 12, 16, 47)
        await self.ask()
        system = self.client.calls[0][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("2026-10-12 16:47", system["content"])
        self.assertIn("星期一", system["content"])

    async def test_time_injection_can_be_disabled(self):
        self.settings.inject_current_time = False
        await self.ask()
        self.assertNotIn("当前真实时间", self.client.calls[0][0]["content"])

    async def test_clear_context_command_empties_history_without_model_call(self):
        await self.ask("第一个问题")
        self.assertTrue(self.responder.history[100])
        calls_before = len(self.client.calls)
        self.now += 31
        await self.ask("清空上下文")
        self.assertEqual(len(self.client.calls), calls_before)
        self.assertNotIn(100, self.responder.history)
        self.assertEqual(self.bot.sent[-1][0], 100)
        self.assertIn("已清空", self.bot.sent[-1][1][0]["data"]["text"])

    async def test_pause_and_resume_commands_control_replies(self):
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        await self.ask("暂停回复")
        self.assertIn(100, self.responder.paused_groups)
        calls_before = len(self.client.calls)
        self.now += 100
        await self.ask("正常提问")
        self.assertEqual(len(self.client.calls), calls_before)
        await self.ask("恢复回复")
        self.assertNotIn(100, self.responder.paused_groups)
        await self.ask("正常提问")
        self.assertEqual(len(self.client.calls), calls_before + 1)

    async def test_status_command_reports_runtime_state(self):
        await self.ask("状态")
        text = self.bot.sent[-1][1][0]["data"]["text"]
        self.assertIn("运行中", text)
        self.assertIn("请求统计", text)
        self.assertEqual(self.client.calls, [])

    def test_drop_reason_names_the_specific_cause(self):
        now = self.now
        key = (100, 200)
        self.assertIsNone(self.responder._drop_reason(100, key, now))

        self.responder.group_until[100] = now + 7
        self.assertIn("本群冷却中", self.responder._drop_reason(100, key, now))
        self.responder.group_until.clear()

        self.responder.user_until[key] = now + 23
        self.assertIn("该用户冷却中", self.responder._drop_reason(100, key, now))
        self.responder.user_until.clear()

        self.responder.busy_groups.add(100)
        self.assertIn("仍在处理中", self.responder._drop_reason(100, key, now))
        self.responder.busy_groups.clear()

        self.settings.max_concurrent_requests = 1
        self.responder.tasks = {object()}
        self.assertIn("并发上限", self.responder._drop_reason(100, key, now))
        self.responder.tasks = set()

    async def test_status_command_shows_remaining_cooldown(self):
        await self.ask()
        self.now += 4
        await self.ask("状态")
        text = self.bot.sent[-1][1][0]["data"]["text"]
        self.assertIn("冷却剩余", text)
        self.assertIn("本群 6s", text)
        self.assertIn("该用户 26s", text)

    async def test_keyword_prefix_is_stripped_before_command_matching(self):
        self.settings.trigger_mode = "keyword"
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        await self.ask("/ai 清空上下文", mentioned=False)
        self.assertEqual(self.client.calls, [])
        self.assertIn("已清空", self.bot.sent[-1][1][0]["data"]["text"])

    async def test_user_scope_isolates_history_between_members(self):
        self.settings.history_scope = "user"
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        await self.ask("甲的问题", user=201)
        await self.ask("乙的问题", user=202)
        second = self.client.calls[1]
        self.assertEqual([m["role"] for m in second], ["system", "user"])
        self.assertEqual(len(self.responder.history[(100, 201)]), 2)
        self.assertEqual(len(self.responder.history[(100, 202)]), 2)

    async def test_commands_bypass_cooldown_but_still_need_trigger(self):
        await self.ask("第一个问题")
        self.now += 1  # 仍在 10 秒群冷却内
        sent_before = len(self.bot.sent)
        await self.ask("状态")
        self.assertEqual(len(self.bot.sent), sent_before + 1)
        self.assertIn("运行中", self.bot.sent[-1][1][0]["data"]["text"])
        await self.ask("状态", mentioned=False)
        self.assertEqual(len(self.bot.sent), sent_before + 1)

    # ===== 拟人化 =====

    async def test_long_reply_waits_then_arrives_in_several_messages(self):
        self.settings.split_threshold_chars = 20
        self.settings.drop_trailing_period = False
        self.client.reply = "第一句差不多够了。第二句也差不多。第三句收个尾吧。"
        await self.ask()
        texts = self.sent_texts()
        self.assertGreater(len(texts), 1)
        self.assertEqual("".join(texts), self.client.reply)
        # 逐条打字：每条发出前都有一次自己的停顿
        self.assertEqual(len(self.slept), len(texts))
        for pause in self.slept:
            self.assertGreaterEqual(pause, self.settings.reply_min_delay_seconds)
            self.assertLessEqual(
                pause,
                self.settings.reply_max_delay_seconds + self.settings.reply_delay_jitter_seconds)

    async def test_legacy_typing_paces_the_whole_reply_once(self):
        # per_segment_typing=false 时退回旧行为：整段算一次，条间用固定间隔
        self.settings.per_segment_typing = False
        self.settings.split_threshold_chars = 20
        self.settings.drop_trailing_period = False
        self.client.reply = "第一句差不多够了。第二句也差不多。第三句收个尾吧。"
        await self.ask()
        texts = self.sent_texts()
        self.assertGreater(len(texts), 1)
        self.assertEqual(len(self.slept), len(texts))
        for gap in self.slept[1:]:
            self.assertGreaterEqual(gap, self.settings.segment_gap_min_seconds)
            self.assertLessEqual(gap, self.settings.segment_gap_max_seconds)

    async def test_short_reply_is_one_message_after_one_pause(self):
        self.client.reply = "嗯，周一"
        await self.ask()
        self.assertEqual(len(self.bot.sent), 1)
        self.assertEqual(len(self.slept), 1)
        self.assertGreaterEqual(self.slept[0], self.settings.reply_min_delay_seconds)

    async def test_markdown_and_assistant_tone_never_reach_the_group(self):
        self.client.reply = ("作为一个AI助手，我很高兴为你解答。\n"
                             "**今天是周一**，10月12号。\n"
                             "- 建议你早点休息\n"
                             "希望对你有所帮助。")
        await self.ask()
        text = "".join(self.sent_texts())
        for junk in ("作为一个AI", "很高兴", "**", "- 建议", "希望对你有所帮助", "您"):
            self.assertNotIn(junk, text)
        self.assertIn("今天是周一", text)
        self.assertIn("10月12号", text)

    async def test_humanization_can_be_turned_off(self):
        self.settings.humanize_replies = False
        self.settings.split_replies = False
        self.client.reply = "**加粗**"
        await self.ask()
        self.assertEqual(self.sent_texts(), ["**加粗**"])

    async def test_at_sender_is_only_added_to_the_first_message(self):
        self.settings.at_sender = True
        self.settings.split_threshold_chars = 10
        self.settings.drop_trailing_period = False
        self.client.reply = "第一句话在这里。第二句话在这里。第三句话在这里。"
        await self.ask()
        self.assertGreater(len(self.bot.sent), 1)
        self.assertEqual(self.bot.sent[0][1][0]["type"], "at")
        for _, segments in self.bot.sent[1:]:
            self.assertEqual(segments[0]["type"], "text")

    async def test_segmented_reply_is_stored_as_one_history_entry(self):
        self.settings.split_threshold_chars = 20
        self.settings.drop_trailing_period = False
        self.client.reply = "第一句差不多够了。第二句也差不多。第三句收个尾吧。"
        await self.ask()
        assistant = [entry[1]["content"] for entry in self.responder.history[100]
                     if entry[1]["role"] == "assistant"]
        # 分几条发出去，但上下文里只算一条回答
        self.assertEqual(len(assistant), 1)
        self.assertEqual(assistant[0].replace("\n", ""), self.client.reply)

    async def test_reply_containing_code_is_sent_as_one_message(self):
        self.settings.drop_trailing_period = False
        self.client.reply = ("写法是这样：\n```python\n# 注释\n"
                             "def f(x):\n    return x + 1\n```\n就这样")
        await self.ask()
        self.assertEqual(len(self.bot.sent), 1)
        text = self.sent_texts()[0]
        self.assertNotIn("```", text)
        self.assertIn("def f(x):", text)
        self.assertIn("    return x + 1", text)

    async def test_failed_segment_stops_the_remaining_ones(self):
        self.settings.split_threshold_chars = 20
        self.settings.drop_trailing_period = False
        self.client.reply = "第一句差不多够了。第二句也差不多。第三句收个尾吧。"
        calls = {"count": 0}
        original = self.bot.send_group

        async def flaky(segments, *, group_id):
            calls["count"] += 1
            if calls["count"] == 2:
                return {"status": "failed", "retcode": 100}
            return await original(segments, group_id=group_id)

        self.bot.send_group = flaky
        await self.ask()
        self.assertEqual(calls["count"], 2)
        self.assertEqual(len(self.bot.sent), 1)
        # 已发出的那部分仍计入上下文，避免下次答非所问
        self.assertTrue(self.responder.history[100][-1][1]["content"])

    # ===== 拟人化（进阶）=====

    async def test_speaker_name_is_used_instead_of_the_qq_number(self):
        event = message("今天好冷")
        event.raw["sender"] = {"card": "小明", "nickname": "小明昵称"}
        await self.responder.handle(event, self.bot)
        await self.settle()
        self.assertIn("小明: 今天好冷", self.client.calls[0][-1]["content"])
        self.assertNotIn("QQ 200", self.client.calls[0][-1]["content"])

    async def test_speaker_name_can_be_turned_off(self):
        self.settings.use_sender_name = False
        event = message("今天好冷")
        event.raw["sender"] = {"card": "小明"}
        await self.responder.handle(event, self.bot)
        await self.settle()
        self.assertIn("QQ 200: 今天好冷", self.client.calls[0][-1]["content"])

    async def test_consecutive_messages_from_one_person_are_merged(self):
        self.settings.memory_mode = "group"
        self.settings.merge_consecutive_messages = True
        await self.ask("第一句", mentioned=False)
        await self.ask("第二句", mentioned=False)
        # 同一人连发两条，合成一条记录，不挤占历史配额
        self.assertEqual(len(self.responder.history[100]), 1)
        self.assertIn("第一句", self.responder.history[100][0][1]["content"])
        self.assertIn("第二句", self.responder.history[100][0][1]["content"])

    async def test_different_people_are_not_merged(self):
        self.settings.memory_mode = "group"
        await self.ask("甲说的", mentioned=False, user=201)
        await self.ask("乙说的", mentioned=False, user=202)
        self.assertEqual(len(self.responder.history[100]), 2)

    async def test_merging_can_be_turned_off(self):
        self.settings.memory_mode = "group"
        self.settings.merge_consecutive_messages = False
        await self.ask("第一句", mentioned=False)
        await self.ask("第二句", mentioned=False)
        self.assertEqual(len(self.responder.history[100]), 2)

    async def test_merge_window_expiry_keeps_messages_separate(self):
        self.settings.memory_mode = "group"
        self.settings.merge_window_seconds = 30
        await self.ask("第一句", mentioned=False)
        self.now += 31
        await self.ask("很久以后", mentioned=False)
        self.assertEqual(len(self.responder.history[100]), 2)

    async def test_mood_is_injected_and_can_be_disabled(self):
        self.responder.time_provider = lambda: datetime(2026, 10, 13, 3, 0)
        await self.ask()
        system = self.client.calls[0][0]["content"]
        self.assertIn("你现在的状态", system)
        self.assertIn("睡", system)

        self.settings.mood_enabled = False
        self.now += 100
        await self.ask()
        self.assertNotIn("你现在的状态", self.client.calls[1][0]["content"])

    async def test_repeated_answer_is_rewritten_once(self):
        self.settings.anti_repeat_replies = 5
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        repeated = "今天挺冷的，出门记得多穿点衣服。"
        self.client.reply = repeated
        await self.ask("今天冷吗")
        calls_after_first = len(self.client.calls)
        self.assertEqual(self.responder.stats["rewrites"], 0)
        # 第二次问同样的问题，模型又想给一模一样的回答；
        # 被要求换说法之后，这次给出了不同的答案
        self.client.reply = repeated
        self.client.rewrites = iter(["冷，多穿点吧"])
        await self.ask("今天冷吗")
        # 检测到重复 → 带上"别重复"的提醒重写一次
        self.assertEqual(len(self.client.calls), calls_after_first + 2)
        self.assertIn("不要重复", self.client.calls[-1][0]["content"])
        self.assertEqual(self.responder.stats["rewrites"], 1)
        self.assertEqual(self.sent_texts()[-1], "冷，多穿点吧")

    async def test_rewrite_falls_back_to_the_original_when_still_repeating(self):
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        self.client.reply = "今天挺冷的，出门记得多穿点衣服。"
        await self.ask("今天冷吗")
        # 重写之后还是同一句：不能无限重试，直接用原文发出去
        await self.ask("今天冷吗")
        self.assertEqual(self.responder.stats["rewrites"], 0)
        self.assertIn("今天挺冷的", self.sent_texts()[-1])

    async def test_rewrite_falls_back_to_the_original_on_failure(self):
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        self.client.reply = "今天挺冷的，出门记得多穿点衣服。"
        await self.ask("今天冷吗")
        # 重写那一次请求失败：不能让这条回复彻底消失，还是要把原文发出去
        self.client.rewrite_error = AIError("模拟失败")
        await self.ask("今天冷吗")
        self.assertIn("今天挺冷的", self.sent_texts()[-1])

    async def test_short_repeats_are_allowed(self):
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        self.client.reply = "嗯"
        await self.ask("在吗")
        calls = len(self.client.calls)
        await self.ask("在吗")
        # "嗯"这种短回复重复很正常，不该为它多调一次模型
        self.assertEqual(len(self.client.calls), calls + 1)
        self.assertEqual(self.responder.stats["rewrites"], 0)

    async def test_anti_repeat_can_be_disabled(self):
        self.settings.anti_repeat_replies = 0
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        self.client.reply = "今天挺冷的，出门记得多穿点衣服。"
        await self.ask("今天冷吗")
        calls = len(self.client.calls)
        await self.ask("今天冷吗")
        self.assertEqual(len(self.client.calls), calls + 1)

    async def test_each_segment_gets_its_own_typing_pause(self):
        self.settings.per_segment_typing = True
        self.settings.split_threshold_chars = 15
        self.settings.drop_trailing_period = False
        self.client.reply = "第一句短。第二句也短。第三句也短。"
        await self.ask()
        # 每条都有自己的停顿，且逐条按各自长度算 → 短句停得比长句少
        self.assertEqual(len(self.slept), len(self.sent_texts()))
        self.assertGreater(len(self.slept), 1)

    async def test_missed_mention_is_answered_after_the_cooldown(self):
        self.settings.catch_up_missed = True
        self.settings.group_cooldown_seconds = 5
        self.settings.user_cooldown_seconds = 0
        await self.ask("第一个问题")
        sent_before = len(self.bot.sent)
        # 冷却期内 @ 她：这次不回，但会被记下来。
        # 这里用 handle 而不是 ask，是为了在"补回任务跑起来"之前先看一眼状态。
        await self.responder.handle(message("第二个问题"), self.bot)
        self.assertEqual(len(self.bot.sent), sent_before)
        self.assertIn(100, self.responder.missed)
        # 补回任务在等本群冷却，睡够了就自己把这条补上（不需要群里再有人说话）
        await self.settle()
        self.assertNotIn(100, self.responder.missed)
        self.assertEqual(self.responder.stats["catch_up"], 1)
        self.assertEqual(len(self.bot.sent), sent_before + 1)

    async def test_stale_missed_mention_is_pruned(self):
        self.settings.catch_up_ttl_seconds = 30
        self.responder.missed[100] = {"user": 200, "text": "太旧了", "speaker": "QQ 200", "at": self.now}
        self.responder._prune(self.now + 29)
        self.assertIn(100, self.responder.missed)
        # 超过 TTL 的旧消息该丢掉：补了也接不上话
        self.responder._prune(self.now + 31)
        self.assertNotIn(100, self.responder.missed)

    async def test_catch_up_does_not_fire_when_disabled(self):
        self.settings.catch_up_missed = False
        self.settings.group_cooldown_seconds = 5
        self.settings.user_cooldown_seconds = 0
        await self.ask("第一个问题")
        await self.responder.handle(message("第二个问题"), self.bot)
        self.assertEqual(self.responder.missed, {})
        self.assertEqual(self.responder.stats["catch_up"], 0)

    async def test_catch_up_is_skipped_while_paused(self):
        self.settings.catch_up_missed = True
        self.settings.group_cooldown_seconds = 5
        self.settings.user_cooldown_seconds = 0
        await self.ask("第一个问题")
        await self.responder.handle(message("第二个问题"), self.bot)
        self.assertIn(100, self.responder.missed)
        # 群主明确暂停了，就不该偷偷补发
        self.responder.paused_groups.add(100)
        await self.settle()
        self.assertEqual(self.responder.stats["catch_up"], 0)

    async def test_proactive_chatter_is_off_by_default(self):
        self.assertEqual(self.settings.proactive_enabled, False)
        await self.ask("群里在聊今天好冷啊", mentioned=False)
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.responder.stats["proactive"], 0)

    async def test_proactive_chatter_respects_cooldown_and_relevance(self):
        self.settings.proactive_enabled = True
        self.settings.proactive_chance = 1.0
        self.settings.proactive_cooldown_seconds = 300
        self.settings.memory_mode = "group"
        # 先垫一句上文，让她有话题可以接
        await self.ask("今天真的好冷啊", mentioned=False)
        self.assertEqual(self.responder.stats["proactive"], 0)
        await self.ask("今天真的好冷，冻死了", mentioned=False)
        self.assertEqual(self.responder.stats["proactive"], 1)
        # 冷却内不再主动搭话
        self.now += 10
        await self.ask("今天真的好冷，冻死了", mentioned=False)
        self.assertEqual(self.responder.stats["proactive"], 1)

    async def test_proactive_chatter_skips_unrelated_small_talk(self):
        self.settings.proactive_enabled = True
        self.settings.proactive_chance = 1.0
        self.settings.memory_mode = "group"
        await self.ask("今天真的好冷啊", mentioned=False)
        await self.ask("哦", mentioned=False)
        # 太短、接不上话题的话不该插嘴
        self.assertEqual(self.responder.stats["proactive"], 0)

    async def test_followup_adds_one_short_message(self):
        self.settings.followup_chance = 1.0
        self.settings.followup_max_chars = 20
        self.client.reply = "今天挺冷的"
        replies = iter(["今天挺冷的", "啊对了，明天更冷"])
        self.client.complete = lambda messages: _next_reply(replies, self.client, messages)
        await self.ask()
        self.assertEqual(self.sent_texts(), ["今天挺冷的", "啊对了，明天更冷"])

    async def test_followup_is_skipped_when_the_model_says_nothing(self):
        self.settings.followup_chance = 1.0
        replies = iter(["今天挺冷的", "无"])
        self.client.complete = lambda messages: _next_reply(replies, self.client, messages)
        await self.ask()
        self.assertEqual(self.sent_texts(), ["今天挺冷的"])

    async def test_followup_is_off_by_default(self):
        self.assertEqual(self.settings.followup_chance, 0.0)
        await self.ask()
        self.assertEqual(len(self.bot.sent), 1)

    async def test_status_reports_the_new_counters(self):
        await self.ask()
        self.now += 100
        await self.ask("状态")
        text = self.bot.sent[-1][1][0]["data"]["text"]
        self.assertIn("拟人化", text)
        self.assertIn("重写", text)

    async def test_recent_catchphrases_are_reported_to_the_model(self):
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        self.responder.persona = Persona(name="小柚", catchphrases=["确实", "啊这"])
        self.client.reply = "确实，今天挺冷的"
        await self.ask("冷吗")
        self.client.reply = "明天更冷"
        await self.ask("明天呢")
        # 上一句用了"确实"，这次要提醒模型别再用了
        self.assertIn("已经用过这些口头禅", self.client.calls[1][0]["content"])
        self.assertIn("确实", self.client.calls[1][0]["content"])

    async def test_catchphrases_are_scoped_per_group(self):
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        self.responder.persona = Persona(name="小柚", catchphrases=["确实"])
        self.client.reply = "确实，今天挺冷的"
        await self.ask("冷吗")
        # 另一个群不该被 A 群用过的口头禅影响
        await self.ask("冷吗", group=101)
        self.assertNotIn("已经用过这些口头禅", self.client.calls[1][0]["content"])

    async def test_catchphrase_note_is_absent_when_nothing_was_used(self):
        self.responder.persona = Persona(name="小柚", catchphrases=["确实"])
        self.client.reply = "今天挺冷的"
        await self.ask("冷吗")
        self.assertNotIn("已经用过这些口头禅", self.client.calls[0][0]["content"])

    # ===== 生图 =====

    async def test_image_commands_are_ignored_when_the_module_is_off(self):
        # 默认没注入 image_client：连"画一张"都不该被当成生图请求
        scene, wants = self.responder._match_image("画一张在咖啡馆")
        self.assertFalse(wants)
        self.assertEqual(scene, "")
        self.assertFalse(self.responder.image_ready)

    async def test_image_request_is_matched_and_scene_extracted(self):
        client = _FakeImageClient()
        # 明确用 knowledge.toml 里那组触发词，默认值只有前三个
        client.settings.commands = ["画一张", "来个自拍", "生成一张",
                                    "发张照片", "发个照片", "看看照片"]
        self.responder.image_client = client
        self.assertTrue(self.responder.image_ready)
        self.assertEqual(self.responder._match_image("画一张在咖啡馆看书"), ("在咖啡馆看书", True))
        # 只说触发词 → 用配置里的默认场景
        self.assertEqual(self.responder._match_image("来个自拍"), ("", True))
        # 语气词不该被当成场景
        self.assertEqual(self.responder._match_image("发张照片看看"), ("", True))
        self.assertEqual(self.responder._match_image("今天午饭吃啥"), ("", False))

    async def test_image_request_sends_a_photo_and_skips_the_model(self):
        self.settings.image_enabled = True
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        client = _FakeImageClient()
        self.responder.image_client = client
        await self.ask("画一张在咖啡馆")
        self.assertEqual(self.client.calls, [])          # 没有调聊天模型
        self.assertEqual(len(self.bot.sent), 1)
        segments = self.bot.sent[0][1]
        self.assertEqual(segments[0]["type"], "image")
        self.assertIn("file://", segments[0]["data"]["file"])
        self.assertEqual(client.scenes, ["在咖啡馆"])
        self.assertEqual(self.responder.stats["images"], 1)
        # 上下文里留一句"发过照片"，下次聊天她才知道自己干了什么
        self.assertTrue(any("照片" in entry[1]["content"]
                            for entry in self.responder.history[100]))

    async def test_natural_image_request_uses_image_service_and_scene(self):
        self.responder.image_client = _FakeImageClient()
        await self.ask("能不能发张你在咖啡馆的自拍呀")
        self.assertEqual(self.responder.image_client.scenes, ["在咖啡馆"])
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.bot.sent[0][1][0]["type"], "image")

    async def test_image_capability_question_and_negation_use_chat(self):
        self.responder.image_client = _FakeImageClient()
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        for text in ("你会画图吗", "不要发自拍"):
            await self.ask(text)
        self.assertEqual(self.responder.image_client.scenes, [])
        self.assertEqual(len(self.client.calls), 2)

    async def test_natural_image_requests_still_require_group_mention(self):
        self.responder.image_client = _FakeImageClient()
        await self.ask("帮我画一张", mentioned=False)
        self.assertEqual(self.responder.image_client.scenes, [])

    async def test_image_natural_matching_can_be_disabled(self):
        self.responder.image_client = _FakeImageClient()
        self.responder.image_client.settings.natural_requests = False
        self.assertEqual(self.responder._match_image("帮我画一张"), ("", False))
        self.assertEqual(self.responder._match_image("画一张在海边"), ("在海边", True))

    async def test_location_chat_then_see_you_sends_photo_in_same_scene(self):
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        self.responder.image_client = _FakeImageClient()
        self.client.reply = "在咖啡馆靠窗坐着呢，刚点了拿铁"
        await self.ask("你在哪")
        self.assertEqual(self.responder.image_client.scenes, [])
        self.client.reply = '{"appropriate": true, "scene": "咖啡馆靠窗坐着，桌上有拿铁"}'
        await self.ask("看看你")
        self.assertEqual(self.responder.image_client.scenes, ["咖啡馆靠窗坐着，桌上有拿铁"])
        context = json.loads(self.client.calls[-1][-1]["content"])
        self.assertEqual(context["request"], "看看你")
        self.assertIn("在咖啡馆", context["conversation"][-1]["content"])
        self.assertEqual(self.bot.sent[-1][1][0]["type"], "image")
        self.assertIn("咖啡馆", self.responder.history[100][-1][1]["content"])

    async def test_see_you_without_context_uses_default_photo(self):
        self.responder.image_client = _FakeImageClient()
        await self.ask("想看看你")
        self.assertEqual(self.responder.image_client.scenes, [""])
        self.assertEqual(self.client.calls, [])

    async def test_inappropriate_photo_context_stays_in_chat(self):
        self.responder.image_client = _FakeImageClient()
        self.responder._remember(100, 200, "assistant", "正在洗澡，不方便拍照", self.now)
        self.client.reply = '{"appropriate": false, "scene": ""}'
        await self.ask("看看你")
        self.assertEqual(self.responder.image_client.scenes, [])
        self.assertIn("不方便发照片", self.sent_texts()[-1])
        self.assertEqual(self.responder.busy_groups, set())

    async def test_photo_context_failure_does_not_invent_a_scene(self):
        self.responder.image_client = _FakeImageClient()
        self.responder._remember(100, 200, "assistant", "在海边散步", self.now)
        self.client.reply = "不是 JSON"
        await self.ask("看看你")
        self.assertEqual(self.responder.image_client.scenes, [])
        self.assertEqual(self.responder.stats["failures"], 1)
        self.assertEqual(self.responder.busy_groups, set())

    async def test_context_photos_can_be_disabled(self):
        self.responder.image_client = _FakeImageClient()
        self.responder.image_client.settings.context_photos = False
        self.responder._remember(100, 200, "assistant", "在海边散步", self.now)
        await self.ask("看看你")
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.responder.image_client.scenes, [""])

    async def test_photo_context_only_includes_requesting_users_conversation(self):
        self.responder.image_client = _FakeImageClient()
        self.responder._remember(100, 201, "assistant", "在别人的健身房", self.now)
        self.responder._remember(100, 200, "assistant", "在家看书", self.now)
        self.client.reply = '{"appropriate": true, "scene": "在家看书"}'
        await self.ask("看看你")
        context = json.loads(self.client.calls[-1][-1]["content"])
        self.assertNotIn("别人的健身房", str(context))
        self.assertEqual(self.responder.image_client.scenes, ["在家看书"])

    async def test_photo_whitelist_is_checked_before_context_model(self):
        self.responder.image_client = _FakeImageClient()
        self.responder.image_client.settings.allowed_users = [999]
        self.responder._remember(100, 200, "assistant", "在家看书", self.now)
        await self.ask("看看你")
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.responder.image_client.scenes, [])

    async def test_see_you_when_busy_does_not_queue_another_photo(self):
        self.responder.image_client = _FakeImageClient()
        self.responder.busy_groups.add(100)
        await self.ask("看看你")
        self.assertEqual(self.responder.image_client.scenes, [])
        self.assertIn("等我", self.sent_texts()[-1])

    async def test_paused_session_does_not_send_photo(self):
        self.responder.image_client = _FakeImageClient()
        self.responder.paused_groups.add(100)
        await self.ask("看看你")
        self.assertEqual(self.responder.image_client.scenes, [])
        self.assertEqual(self.bot.sent, [])

    async def test_image_failure_is_reported_without_a_stack_trace(self):
        self.settings.image_enabled = True
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        client = _FakeImageClient(error=ImageError("今天的 30 张额度用完了", user_safe=True))
        self.responder.image_client = client
        await self.ask("画一张")
        self.assertEqual(self.bot.sent[0][1][0]["data"]["text"], "这次没发出来：今天的 30 张额度用完了")
        self.assertEqual(self.responder.stats["images"], 0)

    async def test_image_send_failure_logs_napcat_details(self):
        self.responder.image_client = _FakeImageClient()
        self.bot.result = {"status": "failed", "retcode": 1200,
                           "message": "file inaccessible", "wording": "cannot read file"}
        with patch("ai_service.logger") as log:
            await self.ask("来个自拍")
        errors = "\n".join(str(call) for call in log.error.call_args_list)
        self.assertIn("生图", errors)
        self.assertIn("1200", errors)
        self.assertIn("file inaccessible", errors)
        self.assertEqual(self.responder.stats["images"], 1)
        self.assertEqual(self.responder.stats["success"], 0)
        self.assertEqual(self.responder.stats["failures"], 1)

    async def test_literal_at_name_does_not_enter_image_branch(self):
        client = _FakeImageClient()
        self.responder.image_client = client
        await self.ask("@waitday 来个自拍", mentioned=False)
        self.assertEqual(client.scenes, [])
        self.assertEqual(self.bot.sent, [])

    async def test_internal_image_error_only_says_try_later(self):
        self.settings.image_enabled = True
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        client = _FakeImageClient(error=ImageError("生图接口返回 401：{\"error\":\"bad key\"}"))
        self.responder.image_client = client
        await self.ask("画一张")
        text = self.bot.sent[0][1][0]["data"]["text"]
        # 接口细节不该出现在群里
        self.assertEqual(text, "这次照片没发出来，等会儿再试")
        self.assertNotIn("401", text)

    async def test_image_failure_still_releases_the_group(self):
        self.settings.image_enabled = True
        self.settings.group_cooldown_seconds = self.settings.user_cooldown_seconds = 0
        self.responder.image_client = _FakeImageClient(error=ImageError("炸了"))
        await self.ask("画一张")
        # busy 没清掉的话这个群就再也不理人了
        self.assertEqual(self.responder.busy_groups, set())

    async def test_image_status_command_reports_quota(self):
        self.settings.image_enabled = True
        client = _FakeImageClient()
        client.settings.daily_limit = 30
        client._used_today = 28
        self.responder.image_client = client
        await self.ask("生图状态")
        text = self.bot.sent[-1][1][0]["data"]["text"]
        self.assertIn("今天还能发 2 张照片", text)
        self.assertEqual(self.client.calls, [])

    async def test_image_status_when_disabled(self):
        await self.ask("生图状态")
        text = self.bot.sent[-1][1][0]["data"]["text"]
        self.assertIn("关着", text)

    async def test_status_command_mentions_image_state(self):
        await self.ask()
        self.now += 100
        await self.ask("状态")
        self.assertIn("生图", self.bot.sent[-1][1][0]["data"]["text"])

    # ===== 角色卡与知识库 =====
    # 注意：这些是 async 测试，必须待在 IsolatedAsyncioTestCase 子类里。
    # 放进普通 unittest.TestCase 的话 unittest 会把协程当普通返回值丢掉，
    # 测试显示"通过"但其实一行都没跑。

    async def test_keyword_hit_answers_without_calling_the_model(self):
        self.settings.drop_trailing_period = False
        self.use_knowledge(KnowledgeEntry(answer="就三条：别刷屏、别发广告、别吵架。",
                                          question="群规是什么", keywords=["群规"]))
        await self.ask("群规")
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.sent_texts(), ["就三条：别刷屏、别发广告、别吵架。"])
        self.assertEqual(self.responder.stats["knowledge"], 1)
        self.assertEqual(self.responder.stats["requests"], 0)

    async def test_weak_hit_passes_knowledge_to_the_model_instead(self):
        self.use_knowledge(KnowledgeEntry(answer="找群主私聊说一声。", question="怎么请假",
                                          keywords=["休假流程"]))
        await self.ask("请假要走什么流程")
        self.assertEqual(len(self.client.calls), 1)
        system = self.client.calls[0][0]["content"]
        self.assertIn("找群主私聊说一声", system)
        self.assertIn("不相关就直接忽略", system)

    async def test_unrelated_question_gets_no_knowledge_context(self):
        self.use_knowledge(KnowledgeEntry(answer="找群主私聊说一声。", question="怎么请假",
                                          keywords=["休假流程"]))
        await self.ask("今天午饭吃啥")
        self.assertEqual(len(self.client.calls), 1)
        self.assertNotIn("找群主私聊说一声", self.client.calls[0][0]["content"])

    async def test_knowledge_can_be_disabled(self):
        self.settings.knowledge_enabled = False
        self.use_knowledge(KnowledgeEntry(answer="直答内容", keywords=["群规"]))
        await self.ask("群规")
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.responder.stats["knowledge"], 0)

    async def test_direct_answer_still_occupies_the_cooldown(self):
        self.use_knowledge(KnowledgeEntry(answer="直答内容", keywords=["群规"]))
        await self.ask("群规")
        self.now += 1  # 仍在群冷却内
        sent_before = len(self.bot.sent)
        await self.ask("群规")
        self.assertEqual(len(self.bot.sent), sent_before)

    async def test_persona_reaches_the_system_prompt(self):
        self.responder.persona = Persona(name="小柚", identity="是个大学生。", likes=["打游戏"])
        await self.ask()
        system = self.client.calls[0][0]["content"]
        self.assertIn("你叫小柚", system)
        self.assertIn("是个大学生", system)
        self.assertIn("打游戏", system)
        self.assertIn(self.settings.system_prompt, system)

    async def test_status_command_reports_knowledge_counters(self):
        self.use_knowledge(KnowledgeEntry(answer="直答内容", keywords=["群规"]))
        self.responder.persona = Persona(name="小柚")
        await self.ask("群规")
        self.now += 100
        await self.ask("状态")
        text = self.bot.sent[-1][1][0]["data"]["text"]
        self.assertIn("小柚", text)
        self.assertIn("知识库直答 1 次", text)
        self.assertIn("知识库：1 条", text)


class PrivateChatTests(unittest.IsolatedAsyncioTestCase):
    """私聊：默认关闭，开了之后不需要 @，且和群聊互不干扰。"""

    async def asyncSetUp(self):
        self.now = 1000
        self.settings = AISettings(allowed_groups=[100], group_cooldown_seconds=5,
                                   user_cooldown_seconds=0, private_cooldown_seconds=4,
                                   catch_up_missed=False)
        self.client = FakeClient()
        self.bot = FakeBot()
        self.slept = []
        self.responder = GroupResponder(self.settings, self.client, lambda: self.now,
                                        self._fake_sleep)

    async def _fake_sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    async def asyncTearDown(self):
        await self.responder.close()

    async def settle(self):
        for _ in range(20):
            if not self.responder.tasks:
                return
            await asyncio.gather(*list(self.responder.tasks), return_exceptions=True)
        self.fail("后台任务一直没有收敛")

    async def ask(self, *args, **kwargs):
        await self.responder.handle(private_message(*args, **kwargs), self.bot)
        await self.settle()

    def private_texts(self):
        return [segments[-1]["data"]["text"] for _, segments in self.bot.private_sent]

    async def test_private_natural_image_request_needs_no_mention(self):
        self.settings.private_enabled = True
        self.responder.image_client = _FakeImageClient()
        await self.ask("能发张你的照片吗")
        self.assertEqual(self.responder.image_client.scenes, [""])
        self.assertEqual(self.bot.private_sent[0][1][0]["type"], "image")
        self.assertEqual(self.client.calls, [])

    async def test_private_location_then_see_you_uses_current_scene(self):
        self.settings.private_enabled = True
        self.settings.private_cooldown_seconds = 0
        self.responder.image_client = _FakeImageClient()
        self.client.reply = "在公园散步呢"
        await self.ask("你在哪")
        self.assertEqual(self.responder.image_client.scenes, [])
        self.client.reply = '{"appropriate": true, "scene": "在公园散步"}'
        await self.ask("看看你")
        self.assertEqual(self.responder.image_client.scenes, ["在公园散步"])
        self.assertEqual(self.bot.private_sent[-1][1][0]["type"], "image")
        self.assertEqual(self.bot.sent, [])

    async def test_private_is_off_by_default(self):
        self.assertFalse(self.settings.private_enabled)
        await self.ask("你好")
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.bot.private_sent, [])

    async def test_private_reply_goes_to_private_not_group(self):
        self.settings.private_enabled = True
        await self.ask("在吗")
        self.assertEqual(self.bot.sent, [])            # 没发到任何群
        self.assertEqual(len(self.bot.private_sent), 1)
        user_id, segments = self.bot.private_sent[0]
        self.assertEqual(user_id, 200)
        self.assertEqual(segments, [{"type": "text", "data": {"text": "你好！"}}])

    async def test_private_needs_no_at_mention(self):
        self.settings.private_enabled = True
        # 私聊里 @ 没有意义，所以不该要求 mention
        self.settings.trigger_mode = "mention"
        await self.ask("不说话也能回")
        self.assertEqual(len(self.client.calls), 1)

    async def test_private_whitelist_blocks_strangers(self):
        self.settings.private_enabled = True
        self.settings.private_allowed_users = [999]
        await self.ask("我是陌生人", user=200)
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.bot.private_sent, [])

    async def test_private_whitelist_allows_listed_user(self):
        self.settings.private_enabled = True
        self.settings.private_allowed_users = [200]
        await self.ask("我在名单里", user=200)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(len(self.bot.private_sent), 1)

    async def test_empty_private_whitelist_means_everyone(self):
        self.settings.private_enabled = True
        self.settings.private_allowed_users = []
        await self.ask("任何人都行", user=123456)
        self.assertEqual(len(self.client.calls), 1)

    async def test_each_private_conversation_has_its_own_memory(self):
        self.settings.private_enabled = True
        await self.ask("甲说的话", user=201)
        await self.ask("乙说的话", user=202)
        self.assertIn(("private", 201), self.responder.history)
        self.assertIn(("private", 202), self.responder.history)
        # 乙的上下文里不应该出现甲说过的话
        first = self.client.calls[0][-1]["content"]
        second = self.client.calls[1][-1]["content"]
        self.assertIn("甲说的话", first)
        self.assertNotIn("甲说的话", second)

    async def test_private_context_is_not_shared_with_group(self):
        self.settings.private_enabled = True
        await self.ask("私聊里的秘密")
        self.now += 100
        await self.responder.handle(message("群里的问题"), self.bot)
        await self.settle()
        group_content = " ".join(m["content"] for m in self.client.calls[-1])
        self.assertNotIn("私聊里的秘密", group_content)

    async def test_private_uses_its_own_cooldown(self):
        self.settings.private_enabled = True
        self.settings.private_cooldown_seconds = 30
        self.settings.group_cooldown_seconds = 5
        await self.ask("第一句")
        calls = len(self.client.calls)
        # 群冷却只有 5 秒，但私聊冷却 30 秒：这里必须还在冷却里
        self.now += 6
        await self.ask("第二句")
        self.assertEqual(len(self.client.calls), calls)

    async def test_private_commands_work(self):
        self.settings.private_enabled = True
        await self.ask("状态")
        text = self.private_texts()[-1]
        self.assertIn("私聊", text)
        self.assertIn("运行中", text)
        self.assertEqual(self.client.calls, [])    # 命令不调模型

    async def test_private_clear_command_wipes_that_conversation(self):
        self.settings.private_enabled = True
        self.settings.private_cooldown_seconds = 0
        await self.ask("记住这句")
        self.assertIn(("private", 200), self.responder.history)
        await self.ask("清空上下文")
        self.assertNotIn(("private", 200), self.responder.history)
        self.assertIn("已清空", self.private_texts()[-1])

    async def test_private_pause_and_resume(self):
        self.settings.private_enabled = True
        self.settings.private_cooldown_seconds = 0
        await self.ask("暂停回复")
        self.assertIn(("private", 200), self.responder.paused_groups)
        calls = len(self.client.calls)
        await self.ask("正常问题")
        self.assertEqual(len(self.client.calls), calls)
        await self.ask("恢复回复")
        await self.ask("正常问题")
        self.assertEqual(len(self.client.calls), calls + 1)

    async def test_private_never_chats_proactively(self):
        # 一对一里"主动搭话"是骚扰，不是像人
        self.settings.private_enabled = True
        self.settings.proactive_enabled = True
        self.settings.proactive_chance = 1.0
        self.settings.memory_mode = "group"
        await self.ask("今天真的好冷啊", user=200)
        await self.ask("今天真的好冷，冻死我了", user=200)
        self.assertEqual(self.responder.stats["proactive"], 0)

    async def test_private_session_helpers(self):
        self.assertTrue(self.responder._is_private(("private", 200)))
        self.assertFalse(self.responder._is_private(100))
        self.assertEqual(self.responder._session_label(("private", 200)), "私聊 200")
        self.assertEqual(self.responder._session_label(100), "群 100")
        # 冷却取值也要分家
        self.settings.private_cooldown_seconds = 7
        self.settings.group_cooldown_seconds = 3
        self.assertEqual(self.responder._cooldown_for(("private", 200)), 7)
        self.assertEqual(self.responder._cooldown_for(100), 3)

    async def test_group_chat_still_needs_a_mention(self):
        # 加了私聊之后，群聊的触发规则不能被放松
        self.settings.trigger_mode = "mention"
        await self.responder.handle(message("没 @ 我", mentioned=False), self.bot)
        await self.settle()
        self.assertEqual(self.client.calls, [])

    async def test_private_knowledge_answer_also_goes_to_private(self):
        self.settings.private_enabled = True
        self.responder.knowledge = KnowledgeBase(
            [KnowledgeEntry(answer="就三条：别刷屏、别发广告、别吵架。",
                            question="群规是什么", keywords=["群规"])])
        await self.ask("群规")
        self.assertEqual(self.client.calls, [])          # 直答不调模型
        self.assertEqual(self.bot.sent, [])              # 也没发到群里
        self.assertIn("别刷屏", self.private_texts()[-1])

    async def test_private_message_id_dedup_is_per_session(self):
        self.settings.private_enabled = True
        self.settings.private_cooldown_seconds = 0
        await self.ask("同一条", message_id=7)
        calls = len(self.client.calls)
        await self.ask("同一条", message_id=7)          # 重复 id 应当被丢掉
        self.assertEqual(len(self.client.calls), calls)

    def test_private_settings_are_validated(self):
        with self.assertRaises(ValueError):
            AISettings(private_allowed_users=["200"])   # 字符串 QQ 号会静默失效
        with self.assertRaises(ValueError):
            AISettings(private_history_messages=-1)
        with self.assertRaises(ValueError):
            AISettings(private_cooldown_seconds=-1)
        with self.assertRaises(ValueError):
            AISettings(private_enabled="yes")
        self.assertTrue(AISettings(private_enabled=True).private_enabled)


class HumanizeHelperTests(unittest.TestCase):
    """反重复、状态、称呼、临时提醒的纯函数部分。"""

    def test_similarity_detects_repeats_but_not_unrelated_text(self):
        self.assertGreater(similarity("今天挺冷的，多穿点", "今天挺冷的，多穿一点"), 0.7)
        self.assertLess(similarity("今天挺冷的，多穿点", "明天要下雨了"), 0.2)
        self.assertEqual(similarity("", "随便什么"), 0.0)

    def test_repetition_check_uses_the_threshold(self):
        recent = ["今天挺冷的，多穿点"]
        self.assertTrue(is_repetitive("今天挺冷的，多穿点", recent, threshold=0.75))
        self.assertFalse(is_repetitive("明天要下雨了", recent, threshold=0.75))

    def test_mood_note_reflects_the_hour(self):
        late = mood_note(datetime(2026, 10, 13, 3, 0))
        evening = mood_note(datetime(2026, 10, 13, 21, 0))
        self.assertIn("睡", late)
        self.assertIn("有精神", evening)
        self.assertIn("工作日", evening)

    def test_mood_note_marks_weekends(self):
        self.assertIn("周末", mood_note(datetime(2026, 10, 17, 11, 0)))

    def test_system_note_merges_into_the_leading_system_message(self):
        messages = [{"role": "system", "content": "原设定"}, {"role": "user", "content": "你好"}]
        merged = with_system_note(messages, "补充提醒")
        # 不新增 system 消息、也不插到中间：个别服务商只接受开头的 system
        self.assertEqual([m["role"] for m in merged], ["system", "user"])
        self.assertIn("原设定", merged[0]["content"])
        self.assertIn("补充提醒", merged[0]["content"])

    def test_system_note_is_added_when_there_is_no_system_message(self):
        merged = with_system_note([{"role": "user", "content": "你好"}], "提醒")
        self.assertEqual([m["role"] for m in merged], ["system", "user"])

    def test_sender_name_prefers_group_card_then_nickname(self):
        carded = message()
        carded.raw["sender"] = {"card": "群名片", "nickname": "昵称"}
        self.assertEqual(sender_name(carded), "群名片")
        nicknamed = message()
        nicknamed.raw["sender"] = {"card": "", "nickname": "昵称"}
        self.assertEqual(sender_name(nicknamed), "昵称")

    def test_sender_name_falls_back_to_the_qq_number(self):
        self.assertEqual(sender_name(message()), "QQ 200")


class HumanizeTests(unittest.TestCase):
    def test_markdown_markers_are_removed_but_text_is_kept(self):
        raw = "## 标题\n**重点**\n- 第一项\n1. 第二项\n`代码`\n---\n> 引用"
        text = humanize_reply(raw, drop_trailing_period=False)
        for junk in ("##", "**", "- 第一项", "1. 第二项", "`", "---", "> "):
            self.assertNotIn(junk, text)
        for kept in ("标题", "重点", "第一项", "第二项", "代码", "引用"):
            self.assertIn(kept, text)

    def test_assistant_filler_is_dropped_but_content_survives(self):
        text = humanize_reply("首先，今天降温了。其次，记得加衣服。希望对你有所帮助。")
        self.assertNotIn("首先", text)
        self.assertNotIn("其次", text)
        self.assertNotIn("希望对你有所帮助", text)
        self.assertIn("今天降温了", text)
        self.assertIn("记得加衣服", text)

    def test_polite_you_and_short_reply_trailing_period(self):
        self.assertEqual(humanize_reply("您好，是的。"), "你好，是的")
        self.assertEqual(humanize_reply("您好，是的。", drop_trailing_period=False), "你好，是的。")

    def test_pure_assistant_talk_falls_back_to_original(self):
        raw = "作为一个AI助手，希望对你有所帮助。"
        self.assertEqual(humanize_reply(raw), raw)

    def test_assistant_closers_are_removed(self):
        for closer in ("有什么我能帮你的吗？", "还有什么可以帮你？", "如果还有其他问题，随时问我。",
                       "希望我的回答对你有帮助。", "很高兴能为你解答。",
                       "有什么工程问题需要帮忙吗？"):
            with self.subTest(closer=closer):
                self.assertEqual(humanize_reply(f"今天降温了。{closer}"), "今天降温了")

    def test_normal_talk_about_helping_is_not_eaten(self):
        for text in ("我帮不了你，你去问问别人吧", "你有什么打算，我可以帮你想"):
            with self.subTest(text=text):
                self.assertIn("帮", humanize_reply(text))

    def test_short_reply_is_not_split(self):
        self.assertEqual(split_reply("嗯，周一", max_segments=3, threshold_chars=60), ["嗯，周一"])

    def test_long_reply_is_split_within_limit_without_losing_text(self):
        raw = "第一句话在这里。第二句话在这里。第三句话在这里。第四句话在这里。"
        parts = split_reply(raw, max_segments=3, threshold_chars=10)
        self.assertEqual(len(parts), 3)
        self.assertEqual("".join(parts), raw)

    def test_unpunctuated_text_is_not_cut_midway(self):
        raw = "啊" * 200
        self.assertEqual(split_reply(raw, max_segments=3, threshold_chars=10), [raw])

    def test_model_inserted_line_breaks_become_separate_messages(self):
        parts = split_reply("第一行很短\n第二行也很短", max_segments=3, threshold_chars=60)
        self.assertEqual(parts, ["第一行很短", "第二行也很短"])

    def test_many_lines_are_sent_whole_so_code_stays_readable(self):
        raw = "```python\ndef f():\n    return 1\n\nprint(f())\n```"
        self.assertEqual(split_reply(raw, max_segments=3, threshold_chars=60), [raw])

    def test_unfenced_code_is_also_detected_and_kept_whole(self):
        raw = ("这么写就行：\ndef qsort(a):\n    if len(a) <= 1:\n        return a\n"
               "    p = a[0]\n    return qsort([x for x in a[1:] if x < p])")
        self.assertEqual(split_reply(raw, max_segments=3, threshold_chars=20), [raw])

    def test_long_prose_with_many_lines_is_still_split(self):
        raw = "\n".join(f"第{index}句讲了一件事情，说得比较长一些。" for index in range(1, 9))
        parts = split_reply(raw, max_segments=3, threshold_chars=40)
        self.assertEqual(len(parts), 3)
        self.assertEqual("".join(parts).replace("\n", ""), raw.replace("\n", ""))

    def test_merged_segments_keep_their_line_breaks(self):
        raw = "短标题\n" + "很长的一句话在这里很长的一句话在这里。" * 3
        parts = split_reply(raw, max_segments=3, threshold_chars=20)
        self.assertGreater(len(parts), 1)
        self.assertIn("\n", parts[-1])
        self.assertEqual("".join(parts).replace("\n", ""), raw.replace("\n", ""))

    def test_code_fence_is_removed_but_code_itself_is_kept(self):
        raw = "给你个写法：\n```python\n# 这是注释\nx = 1  # 行尾注释\n```"
        text = humanize_reply(raw, drop_trailing_period=False)
        self.assertNotIn("```", text)
        self.assertIn("# 这是注释", text)
        self.assertIn("x = 1", text)


class ParsingTests(unittest.TestCase):
    def test_array_mentions_of_others_and_all_do_not_trigger(self):
        for target in ("all", "123", "100010"):
            event = message()
            event.raw["message"][0]["data"]["qq"] = target
            self.assertFalse(message_text(event)[1])

    def test_cq_format_mentions_and_unescaping(self):
        event = message()
        event.raw["message"] = "[CQ:at,qq=10001] 你好 &#91;文本&#93;"
        self.assertEqual(message_text(event), ("你好 [文本]", True))

    def test_invalid_configuration_fails_early(self):
        for kwargs in ({"allowed_groups": ["123"]}, {"trigger_mode": "all"},
                       {"history_messages": -1}, {"request_timeout_seconds": 0},
                       {"keywords": [""]}, {"enabled": "false"},
                       {"history_scope": "channel"}, {"retry_attempts": -1},
                       {"retry_attempts": 1.5}, {"status_commands": [""]},
                       {"inject_current_time": "yes"}, {"temperature": 3},
                       {"temperature": "热"}, {"presence_penalty": 5},
                       {"max_reply_segments": 0}, {"split_threshold_chars": 0},
                       {"humanize_replies": "yes"}, {"drop_trailing_period": "yes"},
                       {"reply_min_delay_seconds": -1}, {"reply_max_delay_seconds": 0.1},
                       {"segment_gap_max_seconds": 0.1}, {"knowledge_enabled": "yes"},
                       {"persona_path": ""}, {"knowledge_path": 1},
                       {"knowledge_direct_threshold": 1.5}, {"knowledge_context_threshold": -0.1},
                       {"knowledge_direct_threshold": 0.1, "knowledge_context_threshold": 0.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AISettings(**kwargs)


class KnowledgeTests(unittest.TestCase):
    def base(self, *entries, direct=0.55, context=0.15):
        return KnowledgeBase(list(entries), direct_threshold=direct, context_threshold=context)

    def test_keyword_match_wins_over_similarity(self):
        kb = self.base(KnowledgeEntry(answer="A", keywords=["群规"]))
        self.assertEqual(kb.lookup("群规")[0], "keyword")

    def test_unrelated_question_scores_zero(self):
        kb = self.base(KnowledgeEntry(answer="A", question="群规是什么", keywords=["群规"]))
        self.assertEqual(kb.lookup("今天午饭吃啥")[0], "none")

    def test_unknown_words_do_not_create_false_matches(self):
        # 修复前：语料里没出现过的词 IDF 取 0，等于说"完全不懂的词毫无信息"，
        # 结果只靠一个公共词"什么"就能匹配到不相干的条目。
        kb = self.base(KnowledgeEntry(answer="A", question="群规是什么"),
                       KnowledgeEntry(answer="B", question="怎么请假"),
                       KnowledgeEntry(answer="C", question="谁做的"))
        level, _, score = kb.lookup("三国演义讲什么")
        self.assertEqual(level, "none")
        self.assertLess(score, 0.15)

    def test_close_paraphrase_reaches_the_context_tier(self):
        kb = self.base(KnowledgeEntry(answer="A", question="怎么请假", keywords=["休假流程"]))
        self.assertEqual(kb.lookup("请假要走什么流程")[0], "weak")

    def test_empty_base_is_always_none(self):
        self.assertEqual(self.base().lookup("群规")[0], "none")

    def test_load_missing_file_returns_empty_base(self):
        self.assertEqual(KnowledgeBase.load(Path("这个文件不存在.toml")).entries, [])

    def test_invalid_entry_is_rejected(self):
        for raw in ({"question": "x"}, {"answer": ""}, {"answer": "x", "keywords": [""]},
                    {"answer": "x", "question": 1}, "不是表"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                KnowledgeEntry.parse(raw)

    def test_snippet_tells_the_model_it_may_ignore(self):
        entry = KnowledgeEntry(answer="A", question="Q")
        self.assertIn("不相关就直接忽略", KnowledgeBase([]).snippet(entry))


class ImageServiceTests(unittest.TestCase):
    """生图模块的纯函数与配置健壮性。不发任何网络请求。"""

    def test_bad_config_types_do_not_crash(self):
        # 手写 TOML 很容易写错类型；写错该掰回来，不该让机器人起不来
        for raw, attr, expected in (
            ({"enabled": "yes"}, "enabled", True),
            ({"enabled": "no"}, "enabled", False),
            ({"daily_limit": "30"}, "daily_limit", 30),
            ({"daily_limit": "abc"}, "daily_limit", 30),
            ({"daily_limit": -5}, "daily_limit", 30),
            ({"cooldown_seconds": "60"}, "cooldown_seconds", 60.0),
            ({"max_refs": "2"}, "max_refs", 2),
            ({"size": ""}, "size", "1728x2304"),
            ({"response_format": "怪东西"}, "response_format", "url"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(getattr(ImageSettings(**raw), attr), expected)

    def test_commands_and_allowed_users_accept_a_bare_string(self):
        # 写成 commands = "画一张" 比数组更常见，不该直接失效
        self.assertEqual(ImageSettings(commands="画一张").commands, ["画一张"])
        self.assertEqual(ImageSettings(allowed_users="123").allowed_users, ["123"])
        self.assertEqual(ImageSettings(commands=[]).commands, ["画一张"])

    def test_allowed_users_empty_means_everyone(self):
        self.assertTrue(ImageSettings().allows(999))
        only_me = ImageSettings(allowed_users=[10001])
        self.assertTrue(only_me.allows(10001))
        self.assertTrue(only_me.allows("10001"))   # 字符串形式也要认
        self.assertFalse(only_me.allows(123))

    def test_clean_scene_strips_cq_and_trims_length(self):
        self.assertEqual(clean_scene("[CQ:at,qq=1]  在咖啡馆   看书")[0], "在咖啡馆 看书")
        self.assertEqual(clean_scene("")[0], "")
        self.assertEqual(clean_scene("啊" * 500)[0], "啊" * 120)

    def test_clean_scene_rejects_obviously_bad_input(self):
        scene, reason = clean_scene("画个裸体")
        self.assertEqual(scene, "")
        self.assertIn("不合适", reason)
        self.assertIsNone(clean_scene("在图书馆看书")[1])

    def test_prompt_puts_the_character_before_the_scene(self):
        # 场景来自群友、不可信，放前面会把人物带跑
        prompt = compose_prompt(FALLBACK_CHARACTER, "在火星上")
        self.assertTrue(prompt.startswith(FALLBACK_CHARACTER["appearance"][:8]))
        self.assertTrue(prompt.rstrip("。").endswith("在火星上"))

    def test_empty_scene_falls_back_to_the_configured_one(self):
        prompt = compose_prompt(FALLBACK_CHARACTER, "")
        self.assertIn(FALLBACK_CHARACTER["scene"], prompt)

    def test_extract_image_handles_both_response_shapes(self):
        self.assertEqual(ImageClient._extract_image({"data": [{"url": "http://x/1.png"}]}),
                         {"kind": "url", "value": "http://x/1.png"})
        self.assertEqual(ImageClient._extract_image({"data": [{"b64_json": "AAA"}]}),
                         {"kind": "b64", "value": "AAA"})

    def test_extract_image_reports_api_errors(self):
        for payload in ({}, {"data": []}, {"data": ["不是字典"]}, ["不是字典"],
                        {"error": {"message": "额度不足"}}):
            with self.subTest(payload=payload):
                with self.assertRaises(ImageError):
                    ImageClient._extract_image(payload)

    def test_quota_and_cooldown_block_reasons(self):
        client = ImageClient(FALLBACK_CHARACTER, ImageSettings(enabled=True, api_key="k",
                                                              daily_limit=1, cooldown_seconds=60))
        self.assertIsNone(client.blocked_reason(0))
        client._used_today = 1
        self.assertIn("额度用完", client.blocked_reason(0))

        fresh = ImageClient(FALLBACK_CHARACTER, ImageSettings(enabled=True, api_key="k",
                                                             daily_limit=5, cooldown_seconds=60))
        fresh._last_at = 100.0
        self.assertIn("冷却中", fresh.blocked_reason(130))
        self.assertIsNone(fresh.blocked_reason(200))

    def test_clock_starting_at_zero_still_counts_as_a_first_call(self):
        # _last_at 用 None 而不是 0：clock 从 0 开始时，0 会被误判成"刚出过图"
        client = ImageClient(FALLBACK_CHARACTER, ImageSettings(enabled=True, api_key="k",
                                                              cooldown_seconds=60))
        self.assertIsNone(client.blocked_reason(0))

    def test_reserve_is_atomic_and_release_returns_the_quota(self):
        client = ImageClient(FALLBACK_CHARACTER, ImageSettings(enabled=True, api_key="k",
                                                              daily_limit=1, cooldown_seconds=0))
        self.assertIsNone(client._reserve(0))
        # 第二个并发请求必须被挡住，否则两个都通过就把额度打穿了
        self.assertIn("额度用完", client._reserve(0))
        client._release()
        self.assertIsNone(client._reserve(0))

    def test_output_paths_do_not_collide_within_one_second(self):
        # 不用 TemporaryDirectory/mkdtemp：它们在创建或清理时会 chmod，
        # 在受限沙箱里会被拒。这里用工作区里一个普通目录。
        temp = Path(".tmp/test_output_paths")
        temp.mkdir(parents=True, exist_ok=True)
        client = ImageClient(FALLBACK_CHARACTER,
                             ImageSettings(enabled=True, api_key="k", output_dir=str(temp)))
        first = client._output_path()
        first.write_bytes(b"x")
        second = client._output_path()
        # 只用秒级时间戳的话这里会撞上，第二张把第一张冲掉
        self.assertNotEqual(first, second)

    def test_selftest_config_reports_without_a_key(self):
        settings = ImageSettings()
        self.assertFalse(settings.configured)
        self.assertIn("未配置", settings.describe())
        self.assertNotIn("api_key", settings.describe())

    def test_build_image_segment_uses_a_local_file_uri(self):
        segment = build_image_segment("generated/x.png")
        self.assertEqual(segment["type"], "image")
        self.assertTrue(segment["data"]["file"].startswith("file://"))

    def test_references_are_sorted_naturally_not_as_strings(self):
        # 纯字符串排序会把 "(10)" 排在 "(2)" 前面，于是"取前 N 张"拿到的是
        # (1)(10)(11) —— 完全不是用户按文件名排的那个优先级。
        names = ["1.png", "2.png", "10.png", "11.png"]
        self.assertEqual(sorted(names, key=lambda n: _natural_key(Path(n))),
                         ["1.png", "2.png", "10.png", "11.png"])
        # 对照：纯字符串排序是错的
        self.assertEqual(sorted(names), ["1.png", "10.png", "11.png", "2.png"])
        # 数字相同时（02 与 2）退回比较后面的文字，顺序保持稳定
        self.assertEqual(
            sorted(["10_全身.jpg", "2_背面.jpg", "02_侧面.jpg", "01_正面.jpg"],
                   key=lambda n: _natural_key(Path(n))),
            ["01_正面.jpg", "02_侧面.jpg", "2_背面.jpg", "10_全身.jpg"])

    def test_collect_references_picks_in_natural_order(self):
        temp = Path(".tmp/test_refs_order")
        temp.mkdir(parents=True, exist_ok=True)
        for name in ("1.png", "2.png", "10.png", "11.png"):
            (temp / name).write_bytes(b"x")
        picked = [p.name for p in collect_references(str(temp), 3)]
        # 取前 3 张应当是 1,2,10 —— 而不是字符串排序的 1,10,11
        self.assertEqual(picked, ["1.png", "2.png", "10.png"])

    def test_collect_references_ignores_non_images_and_missing_dir(self):
        temp = Path(".tmp/test_refs_types")
        temp.mkdir(parents=True, exist_ok=True)
        (temp / "note.txt").write_text("x", encoding="utf-8")
        (temp / "pic.PNG").write_bytes(b"x")
        self.assertEqual([p.name for p in collect_references(str(temp), 5)], ["pic.PNG"])
        self.assertEqual(collect_references("这个目录不存在", 3), [])
        self.assertEqual(collect_references(str(temp), 0), [])

    def test_reference_images_are_downscaled_before_upload(self):
        # 4K 大图直接 Base64 会撑出十几 MB 请求体；上传前必须缩
        from io import BytesIO
        temp = Path(".tmp/test_refs_resize")
        temp.mkdir(parents=True, exist_ok=True)
        big = temp / "big.png"
        Image.new("RGB", (3000, 4000), (200, 180, 170)).save(big)

        uri = encode_image(big, max_edge=1000, quality=85)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        raw = base64.b64decode(uri.split(",", 1)[1])
        with Image.open(BytesIO(raw)) as shrunk:
            self.assertEqual(max(shrunk.size), 1000)
            self.assertLess(shrunk.size[0] * shrunk.size[1], 3000 * 4000)
        self.assertLess(len(uri), big.stat().st_size)

    def test_reference_downscaling_can_be_disabled(self):
        from io import BytesIO
        temp = Path(".tmp/test_refs_noshrink")
        temp.mkdir(parents=True, exist_ok=True)
        small = temp / "small.png"
        Image.new("RGB", (400, 600), (10, 20, 30)).save(small)
        uri = encode_image(small, max_edge=0)
        # max_edge=0 表示不压缩，应当原样发出 PNG
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        raw = base64.b64decode(uri.split(",", 1)[1])
        with Image.open(BytesIO(raw)) as original:
            self.assertEqual(original.size, (400, 600))

    def test_small_reference_is_not_bloated_by_conversion(self):
        # 小图转 JPEG 可能反而更大，那种情况应当保留原图
        temp = Path(".tmp/test_refs_small")
        temp.mkdir(parents=True, exist_ok=True)
        tiny = temp / "tiny.png"
        Image.new("RGB", (8, 8), (1, 2, 3)).save(tiny)
        uri = encode_image(tiny, max_edge=1536, quality=88)
        self.assertLessEqual(len(uri), len(tiny.read_bytes()) * 4 // 3 + 64)

    def test_ref_compression_settings_are_validated(self):
        self.assertEqual(ImageSettings(ref_max_edge="1000").ref_max_edge, 1000)
        self.assertEqual(ImageSettings(ref_max_edge=-5).ref_max_edge, 1536)
        self.assertEqual(ImageSettings(ref_max_edge=0).ref_max_edge, 0)
        # 质量上限 95：再高只是白白变大
        self.assertEqual(ImageSettings(ref_jpeg_quality=100).ref_jpeg_quality, 95)
        self.assertEqual(ImageSettings(ref_jpeg_quality=0).ref_jpeg_quality, 88)


class PersonaTests(unittest.TestCase):
    def test_empty_fields_are_skipped(self):
        text = Persona(name="小柚", identity="是大学生。", personality=["", "话少"]).describe()
        self.assertIn("你叫小柚", text)
        self.assertIn("话少", text)
        self.assertNotIn("背景", text)
        self.assertNotIn("口头禅", text)

    def test_empty_persona_describes_nothing(self):
        self.assertEqual(Persona().describe(), "")

    def test_load_missing_file_is_empty(self):
        self.assertEqual(Persona.load(Path("这个文件不存在.toml")).name, "")

    def test_self_intro_becomes_an_instant_answer(self):
        kb = build_knowledge(Persona(name="小柚", self_intro="我是小柚。"), KnowledgeBase([]))
        level, entry, _ = kb.lookup("你是谁")
        self.assertEqual(level, "keyword")
        self.assertEqual(entry.answer, "我是小柚。")

    def test_build_knowledge_without_intro_returns_the_same_base(self):
        original = KnowledgeBase([])
        self.assertIs(build_knowledge(Persona(), original), original)


class NetworkErrorTests(unittest.TestCase):
    def test_each_failure_kind_gets_its_own_message(self):
        cases = (
            (asyncio.TimeoutError(), "超时"),
            (aiohttp.InvalidURL("example.com"), "格式不合法"),
            (aiohttp.ClientError("boom"), "网络请求失败"),
            (RuntimeError("boom"), "未预期的错误"),
        )
        for exc, expected in cases:
            with self.subTest(exc=type(exc).__name__):
                self.assertIn(expected, describe_network_error(exc))

    def test_timeout_message_names_the_configured_limit(self):
        detail = describe_network_error(asyncio.TimeoutError(), timeout_seconds=20)
        self.assertIn("20 秒", detail)
        self.assertIn("调大", detail)


class ClientIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.status = 200
        self.statuses = []
        self.data = {"choices": [{"message": {"content": "模型回答"}}]}
        self.delay = 0
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.completions)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.base_url = f"http://127.0.0.1:{self.runner.addresses[0][1]}/v1"
        self.env = {"AI_BASE_URL": self.base_url, "AI_API_KEY": "test-secret", "AI_MODEL": "test-model"}

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def completions(self, request):
        self.requests.append((request.headers.get("Authorization"), await request.json()))
        if self.delay:
            await asyncio.sleep(self.delay)
        status = self.statuses.pop(0) if self.statuses else self.status
        return web.json_response(self.data, status=status)

    def client(self, settings=None):
        with patch.dict(os.environ, self.env):
            return ChatClient(settings or AISettings())

    async def test_http_request_schema(self):
        messages = [{"role": "user", "content": "测试"}]
        answer = await self.client().complete(messages)
        self.assertEqual(answer, "模型回答")
        auth, payload = self.requests[0]
        self.assertEqual(auth, "Bearer test-secret")
        self.assertEqual(payload, {"model": "test-model", "messages": messages, "stream": False,
                                   "max_tokens": 300, "temperature": 1.0, "presence_penalty": 0.3})

    async def test_sampling_parameters_follow_settings(self):
        settings = AISettings(temperature=1.3, presence_penalty=0, frequency_penalty=-0.5)
        await self.client(settings).complete([{"role": "user", "content": "hi"}])
        payload = self.requests[0][1]
        self.assertEqual(payload["temperature"], 1.3)
        self.assertEqual(payload["frequency_penalty"], -0.5)
        # 0 表示不干预，不应把该参数发给服务商
        self.assertNotIn("presence_penalty", payload)

    async def test_errors_and_invalid_schema_are_sanitized(self):
        self.status = 401
        self.data = {"error": "test-secret"}
        with self.assertRaisesRegex(AIError, "HTTP 401") as error:
            await self.client().complete([])
        self.assertNotIn("test-secret", str(error.exception))
        self.status = 200
        with self.assertRaisesRegex(AIError, "格式不兼容"):
            await self.client().complete([])

    async def test_timeout(self):
        self.delay = 0.1
        settings = AISettings(request_timeout_seconds=0.02, retry_attempts=0)
        with self.assertRaisesRegex(AIError, "超时"):
            await self.client(settings).complete([])

    async def test_upstream_failure_has_actionable_error(self):
        self.status = 502
        self.data = {"error": {"type": "upstream_error", "message": "test-secret"}}
        with self.assertRaisesRegex(AIError, "上游不可用.*HTTP 502") as error:
            await self.client().complete([])
        self.assertNotIn("test-secret", str(error.exception))

    async def test_transient_5xx_is_retried_then_succeeds(self):
        self.statuses = [502, 200]
        settings = AISettings(retry_attempts=1, retry_backoff_seconds=0)
        answer = await self.client(settings).complete([{"role": "user", "content": "hi"}])
        self.assertEqual(answer, "模型回答")
        self.assertEqual(len(self.requests), 2)

    async def test_auth_error_is_not_retried(self):
        self.statuses = [401, 200]
        settings = AISettings(retry_attempts=3, retry_backoff_seconds=0)
        with self.assertRaisesRegex(AIError, "鉴权失败.*HTTP 401"):
            await self.client(settings).complete([])
        self.assertEqual(len(self.requests), 1)

    async def test_404_is_not_retried_because_the_path_is_usually_wrong(self):
        self.statuses = [404, 200]
        settings = AISettings(retry_attempts=1, retry_backoff_seconds=0)
        with self.assertRaisesRegex(AIError, "路径不存在.*HTTP 404"):
            await self.client(settings).complete([])
        self.assertEqual(len(self.requests), 1)

    async def test_retry_gives_up_after_the_configured_attempts(self):
        self.status = 502
        settings = AISettings(retry_attempts=2, retry_backoff_seconds=0)
        with self.assertRaisesRegex(AIError, "上游不可用"):
            await self.client(settings).complete([])
        self.assertEqual(len(self.requests), 3)

    async def test_probe_reports_success_and_failure(self):
        ok, detail = await self.client().probe()
        self.assertTrue(ok)
        self.assertIn("HTTP 200", detail)
        self.statuses = [502]
        ok, detail = await self.client().probe()
        self.assertFalse(ok)
        self.assertIn("上游不可用", detail)

    async def test_probe_without_configuration(self):
        with patch.dict(os.environ, {"AI_BASE_URL": "", "AI_API_KEY": "", "AI_MODEL": ""}):
            ok, detail = await ChatClient(AISettings()).probe()
        self.assertFalse(ok)
        self.assertIn("未配置", detail)

    async def test_probe_reports_the_actual_network_cause(self):
        # 127.0.0.1:9 上没有任何服务，会立刻连接被拒
        with patch.dict(os.environ, {**self.env, "AI_BASE_URL": "http://127.0.0.1:9/v1"}):
            ok, detail = await ChatClient(AISettings(probe_timeout_seconds=3)).probe()
        self.assertFalse(ok)
        self.assertIn("无法连接服务商", detail)
        self.assertNotIn("网络不可达", detail)

    async def test_probe_warns_when_the_model_returns_no_text(self):
        self.data = {"choices": [{"finish_reason": "length",
                                  "message": {"content": "", "reasoning_content": "想一下"}}]}
        ok, detail = await self.client().probe()
        self.assertTrue(ok)
        self.assertIn("max_output_tokens", detail)

    async def test_reasoning_model_truncated_output_is_explained(self):
        self.data = {"choices": [{"finish_reason": "length",
                                  "message": {"content": "", "reasoning_content": "想一下"}}]}
        settings = AISettings(max_output_tokens=300)
        with self.assertRaisesRegex(AIError, "max_output_tokens") as error:
            await self.client(settings).complete([])
        self.assertIn("300", str(error.exception))
        self.assertIn("截断", str(error.exception))

    async def test_reasoning_content_without_answer_is_reported(self):
        self.data = {"choices": [{"finish_reason": "stop",
                                  "message": {"content": "", "reasoning_content": "想一下"}}]}
        with self.assertRaisesRegex(AIError, "只返回了推理内容"):
            await self.client().complete([])

    async def test_non_dict_response_is_rejected(self):
        self.data = ["not", "a", "dict"]
        with self.assertRaisesRegex(AIError, "格式不兼容"):
            await self.client().complete([])

    async def test_real_bot_plugin_calls_model_and_sends_text_segments(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "ai.toml").write_text("allowed_groups = [100]\n", encoding="utf-8")
            (path / "plugins.toml").write_text('[plugins."bot.ai"]\nenabled = true\n', encoding="utf-8")
            env = {**self.env, "WS_URL": "ws://127.0.0.1:1", "HTTP_URL": "http://127.0.0.1:1",
                   "NAPCAT_TOKEN": "test-napcat", "NAPCAT_WS_TOKEN": "test-ws"}
            with patch.dict(os.environ, env):
                bot = create_bot(plugins_path=path / "plugins.toml", ai_config_path=path / "ai.toml")
            fake_bot = FakeBot()
            bot.send_group = fake_bot.send_group
            # bot.ai 同时注册群聊和私聊入口。
            self.assertEqual(len(bot.handler.get_handlers()), 2)
            await bot._handle_raw_event(message().raw)
            await asyncio.gather(*list(bot.ai_responder.tasks))
            self.assertEqual(len(self.requests), 1)
            self.assertEqual(fake_bot.sent, [(100, [{"type": "text", "data": {"text": "模型回答"}}])])
            await bot.ai_responder.close()
            await bot.adapter.close()
