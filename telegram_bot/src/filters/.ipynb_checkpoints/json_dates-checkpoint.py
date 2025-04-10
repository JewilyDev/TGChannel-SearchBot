from telegram_bot.src.filters.date_actual import extract_dates, replace_temporal_expressions
from datetime import datetime
from db.dbconn import get_all_chunks
from collections import defaultdict
import json


def determinate_today(chunk: dict) -> str:  
    '''
    Определение даты поста, относительно которой считаются другие даты
    '''
    date = []
    
    if chunk['release_date']:
        date = extract_dates(chunk['release_date'])
        if len(date):
            date = date[0]

    if not len(date):
        date = chunk['message_date']

    return datetime.strptime(date, '%d.%m.%Y')


def create_json_dates(chunks: dict) -> None:
    '''
    Создание json-файла вида {date: [id_chunk_1, ..., id_chunk_n]}
    '''
    todays = [determinate_today(chunk) for chunk in chunks]
    
    # Достаем все уникальные даты из текстов чанков
    chunk_texts = [chunk['text'] for chunk in chunks]
    all_dates = [extract_dates(text, today) for text, today in zip(chunk_texts, todays)]

    # Создаем словарь вида {date: [id_chunk_1, ..., id_chunk_n]}
    dates_to_ids = defaultdict(list)
    for chunk, dates, today in zip(chunks, all_dates, todays):
        today_str = today.strftime('%d.%m.%Y')
        dates_to_ids[today_str].append(chunk['id'])
    
        for date in dates:
            if chunk['id'] not in dates_to_ids[date]:
                dates_to_ids[date].append(chunk['id'])
    
    dates_to_ids = {k: sorted(v) for k, v in dates_to_ids.items()}
    with open("dates_.json", "w") as file:
        json.dump(dates_to_ids, file)
