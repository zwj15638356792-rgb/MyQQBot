from asa import on_keyword, Bot
from asa.core.logging import get_logger

logger = get_logger("bot.echo")

@on_keyword("ping")
async def handle_ping(event, bot: Bot):
    logger.info("ping 处理器已触发")
    result = await bot.reply("pong!", at_sender=True)
    if not isinstance(result, dict) or result.get("status") != "ok" or result.get("retcode") != 0:
        logger.error("NapCat 回复失败：{}", result)
    else:
        logger.info("pong! 已发送")
