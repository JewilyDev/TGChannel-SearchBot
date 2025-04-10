import asyncpg
from pathlib import Path
from configuration.config import config_manager

pool = None
db_config = config_manager.get_config('db')

async def create_pool() -> None:
     global pool
     pool = await asyncpg.create_pool(f"postgres://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}")

async def get_pool():
     if pool is None:
          await create_pool()
     return pool


async def close_pool() -> None:
     await pool.close()

def read_query(path: str) -> str:
     full_path = Path(f'/home/znai/tgbot/tgbotsb/db/queries/{path}.sql').resolve()
     with open(full_path, 'r') as sql_file:
          return sql_file.read()

def with_pool(func):
     async def wrapped(*args, **kwargs):
          global pool
          if pool is None:
               pool = await create_pool()

          kwargs['pool'] = pool  
          return_value = await func(*args, **kwargs)
          return return_value

     return wrapped
     
               




     

    