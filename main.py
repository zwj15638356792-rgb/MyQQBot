import asyncio
import os
from pathlib import Path

from asa import Bot
from asa.core.logging import get_logger
from asa.plugin.config import PluginConfigManager
from dotenv import load_dotenv

from connection import NapCatConnection
from ai_service import AISettings, GroupResponder, KnowledgeBase, Persona, build_knowledge


def _env_flag(name, default=True):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name, default):
    try:
        return float(os.getenv(name, "").strip())
    except ValueError:
        return default


class QQBot(Bot):
    async def _run_main(self):
        # 模型连通性自检放到后台，避免阻塞 WebSocket 连接。
        probe = asyncio.create_task(self.ai_responder.self_check())
        try:
            await super()._run_main()
        finally:
            probe.cancel()
            await asyncio.gather(probe, return_exceptions=True)
            await self.ai_responder.close()


def _build_image_client(settings, config_dir, log):
    """按配置决定要不要加载生图模块。关着就返回 None，整条链路不参与。

    这里刻意容错：生图缺依赖或配置写坏都不该让机器人起不来，
    聊天是主功能，生图是附加项。
    """
    if not settings.image_enabled:
        return None
    try:
        from image_service import ImageClient
    except Exception as exc:
        log.error("[生图] 模块加载失败（{}），生图不可用；聊天不受影响", type(exc).__name__)
        return None
    try:
        client = ImageClient.from_config(config_dir / settings.image_config_path)
    except Exception as exc:
        log.error("[生图] 读取配置失败（{}），生图不可用", type(exc).__name__)
        return None
    log.info("[生图] {}", client.settings.describe())
    if not client.settings.api_key:
        log.warning("[生图] 已开启但没配 IMAGE_API_KEY，出图会失败；请在 .env 里填")
    from image_service import collect_references
    ref_dir = client.character.get("ref_dir", "character_refs")
    if not collect_references(ref_dir, client.settings.max_refs):
        log.warning("[生图] 参考图目录 {} 里没有图：人物会不像，建议放 2~3 张照片", ref_dir)
    return client


def create_bot(*, plugins_path=None, ai_config_path=None):
    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env")
    config_path = Path(ai_config_path) if ai_config_path else project_dir / "ai.toml"
    settings = AISettings.load(config_path)
    # 角色卡与知识库按 ai.toml 所在目录解析，方便把整套配置放到别处。
    config_dir = config_path.parent
    persona = Persona.load(config_dir / settings.persona_path)
    knowledge = build_knowledge(persona, KnowledgeBase.load(
        config_dir / settings.knowledge_path,
        direct_threshold=settings.knowledge_direct_threshold,
        context_threshold=settings.knowledge_context_threshold,
    ))
    log = get_logger("main")
    plugin_config = PluginConfigManager(str(plugins_path or project_dir / "plugins.toml"))
    bot = QQBot(plugin_config_manager=plugin_config, auto_discover=False)
    bot.ai_responder = GroupResponder(
        settings, persona=persona, knowledge=knowledge,
        image_client=_build_image_client(settings, config_dir, log),
    )
    bot.adapter = NapCatConnection(
        bot.config.ws_url,
        bot.config.http_url,
        bot.config.token,
        ws_token=os.getenv("NAPCAT_WS_TOKEN") or bot.config.token,
        reconnect=_env_flag("WS_RECONNECT", True),
        reconnect_initial_seconds=_env_float("WS_RECONNECT_INITIAL_SECONDS", 2.0),
        reconnect_max_seconds=_env_float("WS_RECONNECT_MAX_SECONDS", 60.0),
    )
    get_logger("main").info("已注册 {} 个消息处理器", len(bot.handler.get_handlers()))
    if persona.name:
        get_logger("main").info("角色卡：{}（{}）", persona.name, settings.persona_path)
    else:
        get_logger("main").warning("未读到角色卡（{}）：只用 ai.toml 的 system_prompt", settings.persona_path)
    if knowledge.entries:
        get_logger("main").info("知识库：{} 条（{}）", len(knowledge.entries), settings.knowledge_path)
    else:
        get_logger("main").warning("知识库为空（{}）：所有问题都会调用模型", settings.knowledge_path)
    if not settings.allowed_groups:
        get_logger("main").warning("AI 群白名单为空：不会回复任何群；请编辑 ai.toml")
    if not bot.ai_responder.client.configured:
        get_logger("main").warning("AI 接口未配置：请填写 .env 的 AI_BASE_URL / AI_API_KEY / AI_MODEL")
    return bot


if __name__ == "__main__":
    bot = create_bot()
    bot.plugin_config_manager.start_watch()
    try:
        bot.run()
    finally:
        bot.plugin_config_manager.stop_watch()
