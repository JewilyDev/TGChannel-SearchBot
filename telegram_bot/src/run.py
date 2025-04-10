import asyncio
from logs.logger import main_logger
from create_bot import bot, dp
from handlers.init import init_router
from db.dbconn import create_pool

async def main():
    dp.include_router(init_router)
    await bot.delete_webhook(drop_pending_updates=True)

    await create_pool()
    await dp.start_polling(bot)
    main_logger.info("Ботинок завелся")


if __name__ == "__main__":
    asyncio.run(main())
    