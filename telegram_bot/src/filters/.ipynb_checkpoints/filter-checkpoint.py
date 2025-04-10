import aiogram
from aiogram.types import Message

def not_command(message: Message):
    '''
        Это фильтр очень смешной, все кроме проверки на слеш можно убрать, это для теста нужно щас.
        Мб можно в фильтр отдать лямбда-функцию, но я чет не понял как, поэтому вот))
    '''
    return message.text[0] != "/" and message.text != "Дай манки" and message.text != "Дылда"
