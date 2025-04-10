import asyncpg
from db.dbconn import read_query, with_pool


'''
Функции, относящиеся к папке answers
'''
@with_pool
async def write_answer(
    answer_text: str, 
    tg_message_id: str, 
    id_prompt: str, 
    timestamp, 
    pool = None
) -> None:
    
     query = read_query('answers/write_answer')

     async with pool.acquire() as connection:
          await connection.execute(query, answer_text, tg_message_id, id_prompt, timestamp)

@with_pool
async def insert_feedback(feedback: bool, id_message: str, pool = None) -> None:
     query = read_query('answers/insert_feedback')

     async with pool.acquire() as connection:
          await connection.execute(query, feedback, id_message)


