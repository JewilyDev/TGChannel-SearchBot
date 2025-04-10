import asyncpg
from db.dbconn import read_query, with_pool
'''
Функции, относящиеся к папке prompts
'''
@with_pool
async def write_prompts(text: str, id_user: str, id: str, timestamp, pool = None) -> list[str]:
     query_write = read_query('prompts/write_prompts')
     query_get = read_query('prompts/get_prompts_ids_limit')

     async with pool.acquire() as connection:
          await connection.execute(query_write, text, id_user, id, timestamp)
          id = await connection.fetchrow(query_get, 1)
     return dict(id)['id']