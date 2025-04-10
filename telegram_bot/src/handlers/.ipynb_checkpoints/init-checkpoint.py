from aiogram import Router, F
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup 
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from create_bot import bot, dp
from tg_parser.parser import parse_major, get_channel_info_by_link, get_release_by_message
from configuration.config import config_manager
from db.dbconn import write_user, write_messages, write_prompts, insert_feedback, write_answer, get_user_default_search_src, set_user_default_search_src, create_db, drop_db, close_pool, get_all_messages, get_user_data, get_release_only_mode, set_release_only_mode
from filters.filter import not_command
from logs.logger import main_logger
from text_index.src.index_build import create_search_indicies_by_sources, index_search
from datetime import datetime, date
import os
import html
import json
import requests
from openai import OpenAI

text_cfg = config_manager.get_config("text")
init_router = Router()


class Settings(StatesGroup):
    waiting_for_link = State()


def settings_keyboard():
    keyboard_list = [
        [
         InlineKeyboardButton(text=text_cfg['settings_release_only'], callback_data="release_only_set")
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard= keyboard_list)



def feedback_keyboard(like_text = "", dislike_text = "") -> InlineKeyboardButton:
    '''
        Функция для создания клавиатуры фидбека на сообщении, можно передать текст на кнопки.
        По-умолчанию берутся из конфига тексты.
        Колбек-значения захардкожены, не вижу смысла их куда-то выносить
    '''
    if len(like_text) == 0:
        like_text = text_cfg['feedback_like_text']
    if len(dislike_text) == 0:
        dislike_text = text_cfg['feedback_dislike_text'] 
        
    return InlineKeyboardMarkup(
            inline_keyboard=[
                [  
                    InlineKeyboardButton(text=like_text, callback_data="like"),
                    InlineKeyboardButton(text=dislike_text, callback_data="dislike")
                ]
            ]
        )

@init_router.message(Command("settings"))
async def cmd_start(message: Message):
    await message.answer(text="Настрой меня", reply_markup=settings_keyboard())


@init_router.callback_query(F.data == "source_set")
async def handler_source_set(callback: CallbackQuery, state: FSMContext):
    await callback.message.answer(text=text_cfg['settings_search_link_request'])
    await state.set_state(Settings.waiting_for_link)


@init_router.callback_query(F.data == "release_only_set")
async def handler_source_set(callback: CallbackQuery, state: FSMContext):
    user_id = str(callback.message.chat.id)
    user_data = await get_user_data(user_id)
    release_only =  user_data['release_only']
    await set_release_only_mode(user_id, (not release_only))
    message_answer = text_cfg["settings_release_only_yes"] if not release_only else text_cfg["settings_release_only_no"]
    await callback.message.answer(text=message_answer)
    await bot.edit_message_reply_markup(
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        reply_markup=None
    )



@init_router.message(Settings.waiting_for_link)
async def process_link(message: Message, state: FSMContext):
    avaliable_sources = list(src['link'] for src in config_manager.get_config('parser')['channels'])
    user_link = message.text
    try: 
        res = await get_channel_info_by_link(user_link)
    except:
        await message.answer(text=text_cfg["process_link_wrong_src"])
    if res:
        if user_link not in avaliable_sources:
            await message.answer(text=text_cfg["process_link_not_found"])
        else:
            await set_user_default_search_src(str(message.from_user.id), user_link)
            await message.answer(text=text_cfg["process_link_success"])
    await state.clear()
    await bot.edit_message_reply_markup(
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        reply_markup=None
    )



@init_router.message(CommandStart())
async def cmd_start(message: Message):
    '''
        Для теста здоровья бота
    '''
    sender = str(message.from_user.id)
    await write_user(sender,sender)

    welcome_message = text_cfg['welcome_text'].replace("#FIRSTNAME",message.from_user.first_name)
    await message.answer(welcome_message)


@init_router.message(Command("stop"))
async def cmd_stop(message: Message):
    '''
        Потенциально единственное место смерти бота, но это далеко не факт))
    '''
    await close_pool()
    await message.answer("Пока.")
    os._exit(0)


async def prompt_answer(prompt_message : Message):
    prompt_id = await write_prompts(
        text = prompt_message.text, 
        id_user = str(prompt_message.from_user.id),
        id = str(prompt_message.message_id), 
        timestamp = prompt_message.date)
    
    user_id = str(prompt_message.from_user.id)
    query = prompt_message.text
    user_data = await get_user_data(user_id)
    source = user_data['search_default']
    release_only = user_data['release_only']
    if(source is  None):
        source = ""
    result = await index_search(query = query, source = source, topK = 15, release_only = release_only)
    context = '\n'.join(result['result_output'])
    context = context[:8191]
    question = f"Текущая дата: {date.today()}. Ответь на вопрос учитывая текущую дату. {query}. Поиск по релизам:{release_only}." 
    payload = {
            'question': question,
            'context': context
        }
    
    x = requests.post("http://192.168.1.73:8000/answer", json = payload)
    answer = json.loads(x.text)[0]["text"].replace("</think>", "").replace("\n\n", "").replace('\n', '')
    await prompt_message.answer(answer,reply_markup=feedback_keyboard())
    # await write_answer(
    #     answer_text = answer, 
    #     tg_message_id = str(prompt_message.message_id), 
    #     id_prompt = prompt_id[0], 
    #     timestamp = datetime.now())

@init_router.message(F.text[0] != '/')
async def cmd_search(message: Message):
    await prompt_answer(message)


@init_router.message(Command("parse"))
async def cmd_parse(message: Message):
    '''
        Функция для вызова парсилки и записи результата в бд
    '''
    config_parse = config_manager.get_config("parser")
    data = await parse_major(config_parse['channels'])
    await drop_db()
    await create_db()
    await write_messages(data)
    avaliable_sources = list(src['link'] for src in config_manager.get_config('parser')['channels'])
    await create_search_indicies_by_sources(True, True, True, True, avaliable_sources)
    await create_search_indicies_by_sources(True, True, True, False, avaliable_sources, True)


@dp.callback_query(F.data.in_(["like", "dislike"]))
async def process_callback(callback: CallbackQuery, state: FSMContext):
    '''
        Менеджер фидбека, тут удаляем клавиатуру и пишем в базу фидбек.
    '''
    if callback.data == "like":
        await bot.answer_callback_query(callback.id)    
    elif callback.data == "dislike":
        await bot.answer_callback_query(callback.id)

    # С -1 можно сказать костыль, т.к клава приделывается отдельным message-entity
    # Айди сообщения всегда равен айди клавы - 1, поэтому можно и не бить меня.

    await insert_feedback(feedback = callback.data == "like", id_message= str(callback.message.message_id - 1))
    # Вот эта красавица удаляет фидбек(а разговоров то было...)
    await bot.edit_message_reply_markup(
        chat_id=callback.message.chat.id,
        message_id=callback.message.message_id,
        reply_markup=None
    )

@init_router.message(Command("release"))
def find_release(message : Message, command : Command):
    print(get_release_by_message(command.args))