from telethon import TelegramClient
from configuration.config import config_manager

# Глобальная переменная для хранения клиента
tg_client = None
tg_api_config = config_manager.get_config("tg_api")
api_id = tg_api_config['api_id']
api_hash = tg_api_config['api_hash']

async def init_tg_client():
    global tg_client
    if tg_client is None:
        tg_client = TelegramClient('active_session', api_id, api_hash)
        await tg_client.start()
        print("Клиент успешно инициализирован.")

async def get_tg_client():
    if tg_client is None:
        await init_tg_client()
    return tg_client

async def close_tg_client():
    if tg_client is not None:
        await tg_client.disconnect()
        print("Клиент успешно отключен.")
