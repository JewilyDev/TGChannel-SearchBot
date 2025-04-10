from tg_parser.telegram_con import get_tg_client
from datetime import datetime
from logs.logger import main_logger
from configuration.config import config_manager


parser_config = config_manager.get_config('parser')
async def get_messages(tg_client, chat, username : str, is_chat : bool, limit = -1, last_parsed_timestamp : datetime = "") -> list[tuple]:
        ''' 
            Функция для парсинга сообщений по объекту chat(telethon entity),
            сохраняя результат с псевдонимом username
        '''
        messages = []
        counter = 0
        async for message in tg_client.iter_messages(chat, offset_date = last_parsed_timestamp):
            # В случае с каналами, отправитель - канал, иначе - юзер из ТГ
            sender = username if not is_chat else str(message.from_id.user_id)
            if message.text is not None and len(message.raw_text) > 0:
                message_url = "https://t.me/c/" + str(chat.id) + "/" + str(message.id)
                messages.append(
                    (str(message.id), 
                     sender, 
                     message.raw_text,
                     message.date,
                     get_release_by_message(message.raw_text),
                     message_url
                     ))
                counter += 1
            if limit > 0 and counter == limit:
                break    
        return messages


async def parse_major(channels : dict) -> dict:
    '''Большая функция которая парсит все(пока для теста не все) сообщения из каналов и чатов.
       Каналы и чаты задаются в конфиге parse.json
    '''
    tg_client = await get_tg_client()
    output_messages = {}
    main_logger.info("Запустился парсер.")
    for channel in channels:
        link = channel["link"]

        is_chat = channel["type"] == 'chat'
        is_closed_chat = channel['type'] == 'closed_chat'
        chat = None
        main_logger.info(f"Начинаю парсить {link}")

        try:
            # Такой костылёк нужен, т.к если это чат то link это айди чата, и его нужно сделать интом
            chat = await tg_client.get_entity(link)
            msg = await get_messages(tg_client, chat, link, is_chat)
            output_messages[link] = msg
            
        except ValueError as ve:
            main_logger.error(f"Ошибка: {ve} — возможно, неверный ID или отсутствие доступа")
            
        except Exception as e:
            main_logger.error(f"Произошла ошибка: {e}")
            
    main_logger.info(f"Парсинг завершен")
    return output_messages



async def get_channel_info_by_link(link : str):
    '''
        Получаем telethon-entity по инвайт-ссылке
        link - именно инвайт ссылка для чатов, или просто ссылка на канал
    '''
    tg_client = await get_tg_client()
    entity = await tg_client.get_entity(link)
    print(entity)
    return entity

async def get_message_info_by_id(id : int, sender : str):
    tg_client = await get_tg_client()
    chat = await get_channel_info_by_link(sender)
    message = await tg_client.get_messages(chat.id, ids = id)
    return message


def get_release_by_message(message_text : str):
    '''
        Вычленяем по ключевым словам наличие информации о релизах в тексте
    '''
    for keyword in parser_config['release_keywords']:
        if keyword in message_text:
            release_pos = message_text.find(keyword)
            release_end =  message_text.find("\n",release_pos)
            return message_text[release_pos : release_end]
    return None