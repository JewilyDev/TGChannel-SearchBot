from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from configuration.config import config_manager

bot_config = config_manager.get_config("bot")
tg_api_config = config_manager.get_config("tg_api")

# Если почему-то конфигов нет то и бота запускать смысла нет
if (bot_config is not None) and (tg_api_config is not None):

    bot = Bot(token=bot_config['token'], default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())
    
else:
    print("Error: configs are not specified")
    