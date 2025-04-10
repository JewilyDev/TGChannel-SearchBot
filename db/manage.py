import asyncpg
from db.dbconn import read_query, with_pool
'''
Функции, относящиеся к папке db_manage
'''
@with_pool
async def create_db(pool = None) -> None:
     query = read_query('db_manage/create_db')

     async with pool.acquire() as connection:
          await connection.execute(query)


@with_pool
async def drop_db(pool = None) -> None:
     query = read_query('db_manage/drop_db')

     async with pool.acquire() as connection:
          await connection.execute(query)
