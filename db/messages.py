import asyncpg
from db.dbconn import read_query, with_pool
from db.chats import write_chat

'''
Функции, относящиеся к папке messages
'''
@with_pool
async def write_messages(data: list[tuple], pool = None) -> None:
     query = read_query('messages/write_messages')
     query_get = read_query('messages/get_messages_ids_limit')
     limit = len(data)
     ids = []
     for k in data:
          await write_chat(k, k)
          async with pool.acquire() as connection:
               await connection.executemany(query, data[k])
               ids_messages = await connection.fetch(query_get, limit)
               ids.extend(ids_messages)
     return [dict(rec)['id'] for rec in ids]


@with_pool
async def get_all_messages(pool = None) -> list[dict]:

     query = read_query('messages/get_all_messages')

     async with pool.acquire() as connection:
          messages = await connection.fetch(query)
     
     return [dict(message) for message in messages]

@with_pool
async def get_messages_by_ids(ids: list[str], pool = None) -> list[dict]:

     query = read_query('messages/get_messages_by_ids')

     async with pool.acquire() as connection:
          messages = await connection.fetch(query, ids)
     
     return [dict(msg) for msg in messages]

@with_pool
async def get_messages_by_chat(id_chat: str, pool = None) -> list[dict]:
     query = read_query('messages/get_messages_by_chat')

     async with pool.acquire() as connection:
          messages = await connection.fetch(query, id_chat)
     
     return [dict(msg) for msg in messages]
    
@with_pool
async def get_message_by_chunks(id: str, pool = None) -> list[dict]:
     query = read_query('messages/get_messages_by_chunks')

     async with pool.acquire() as connection:
          message = await connection.fetchrow(query, id)
     
     return dict(message)


@with_pool 
async def get_last_id_by_sender(source: str, pool = None):
     query = read_query('messages/get_last_id_by_sender')

     async with pool.acquire() as connection:
          id = await connection.fetchrow(query, source)
          return id


