import asyncpg
from db.dbconn import read_query, with_pool

@with_pool
async def write_chat(id: str, name: str, pool = None) -> None:
     query = read_query('chats/write_chat')
     async with pool.acquire() as connection:
          await connection.execute(query, id, name)