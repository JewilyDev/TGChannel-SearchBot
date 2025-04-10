import asyncpg
from db.dbconn import read_query, with_pool

'''
Функции, относящиеся к папке users
'''
@with_pool
async def get_release_only_mode(user_id : str, pool = None):
     query = read_query('users/get_release_only')
     async with pool.acquire() as connection:
          release_data = await connection.fetch(query, user_id)
     return dict(release_data)['release_only']

@with_pool
async def set_release_only_mode(user_id : str, release_only : bool, pool = None):
     query = read_query('users/set_release_only')
     async with pool.acquire() as connection:
          await connection.execute(query, user_id, release_only)

@with_pool
async def write_user(id: str, name: str, pool = None) -> None:
     query = read_query('users/write_user')
    
     async with pool.acquire() as connection:
          await connection.execute(query, id, name)

@with_pool
async def get_user_data(id : str, pool = None):
    query = read_query('users/get_user_data')
    async with pool.acquire() as connection:
        user = await connection.fetchrow(query, id)
    return dict(user)

@with_pool
async def get_user_default_search_src(id: str, pool = None) -> list[str]:
     query = read_query('users/get_user_default_search_src')

     async with pool.acquire() as connection:
          chat = await connection.fetchrow(query, id)
     
     return dict(chat)


@with_pool
async def set_user_default_search_src(src: str, id: str, pool = None) -> None:
     query = read_query('users/set_user_default_search_src')

     async with pool.acquire() as connection:
          await connection.executemany(query, src, id)
