import asyncpg
from db.dbconn import read_query, with_pool
'''
Функции, относящиеся к папке chunks
'''
@with_pool
async def write_chunks(chunks_and_ids: list[tuple], pool = None) -> list[str]:
     query_write = read_query('chunks/write_chunks')
     query_get = read_query('chunks/get_chunks_ids_limit')
     limit = len(chunks_and_ids)

     async with pool.acquire() as connection:
          await connection.executemany(query_write, chunks_and_ids)
          ids_chunks = await connection.fetch(query_get, limit)
         
     return [dict(rec)['id'] for rec in ids_chunks]

@with_pool
async def insert_embeddings(embeddings: list[tuple], pool = None) -> None:
     query = read_query('chunks/insert_embeddings')

     async with pool.acquire() as connection:
          await connection.executemany(query, embeddings)

@with_pool
async def get_all_chunks(pool = None) -> list[dict]:
     query = read_query('chunks/get_all_chunks')

     async with pool.acquire() as connection:
          chunks = await connection.fetch(query)
     
     return [dict(chunk) for chunk in chunks]

@with_pool
async def get_chunks_by_ids(chunks_ids: int, pool = None) -> list[dict]:
     query = read_query('chunks/get_chunks_by_ids')

     async with pool.acquire() as connection:
          chunk = await connection.fetchrow(query, chunks_ids)
     
     return dict(chunk)

@with_pool
async def get_timestamps_by_chunk_id(chunks_ids: list[dict], pool = None) -> list[dict]:
     query = read_query('chunks/get_timestamps_by_chunk_id')

     async with pool.acquire() as connection:
          chunks = await connection.fetch(query, chunks_ids)
     
     return [dict(chunk) for chunk in chunks]

@with_pool
async def get_chunks_by_chat(id_chat: str, pool = None) -> list[dict]:
     query = read_query('chunks/get_chunks_by_chat')

     async with pool.acquire() as connection:
          chunks = await connection.fetch(query, id_chat)
     
     return [dict(chunk) for chunk in chunks]

@with_pool
async def get_chunks_by_text(text: str, pool = None) -> list[dict]:
     query = read_query('chunks/get_chunks_by_text')
     async with pool.acquire() as connection:
          chunks = await connection.fetch(query, text)
          
     return [dict(chunk) for chunk in chunks]
