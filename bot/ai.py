from asa import Bot, on_group_message, on_private_message


@on_group_message
async def handle_ai(event, bot: Bot):
    await bot.ai_responder.handle(event, bot)


@on_private_message
async def handle_ai_private(event, bot: Bot):
    """私聊入口。是否真的回复由 ai.toml 的 private_enabled 决定。"""
    await bot.ai_responder.handle(event, bot)
