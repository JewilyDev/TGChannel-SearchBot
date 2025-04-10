import json
import os
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
class MultiConfigManager:
    def __init__(self):
        self.configs = {}  # Хранилище конфигов

    def load_config(self, name, file_path):

        old_path = file_path
        file_path = Path(file_path).resolve()
        """
        Загружает конфиг из JSON-файла или YAML-файла и сохраняет его под указанным именем.
        :param name: Удобное имя для конфига
        :param file_path: Путь к файлу
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        if ".yaml" in old_path:
                self.configs[name] = OmegaConf.load(file_path)
        else:
            with open(file_path, 'r') as f:
                self.configs[name] = json.load(f)

    def get_config(self, name):
        """
        Возвращает конфиг по имени в виде словаря.
        :param name: Имя конфига
        :return: Копия конфига (словарь)
        """
        if name not in self.configs:
            raise KeyError(f"Config '{name}' not found")
        return deepcopy(self.configs[name])

    def update_config(self, name, new_config):
        """
        Обновляет конфиг по имени.
        :param name: Имя конфига
        :param new_config: Новый конфиг (словарь)
        """
        if name not in self.configs:
            raise KeyError(f"Config '{name}' not found")
        self.configs[name] = deepcopy(new_config)

    def save_config(self, name, file_path):
        """
        Сохраняет конфиг в JSON-файл.
        :param name: Имя конфига
        :param file_path: Путь для сохранения
        """
        if name not in self.configs:
            raise KeyError(f"Config '{name}' not found")

        with open(file_path, 'w') as f:
            json.dump(self.configs[name], f, indent=4)

    def list_configs(self):
        """
        Возвращает список всех загруженных конфигов.
        :return: Список имен конфигов
        """
        return list(self.configs.keys())

    def remove_config(self, name):
        """
        Удаляет конфиг по имени.
        :param name: Имя конфига
        """
        if name in self.configs:
            del self.configs[name]



config_manager = MultiConfigManager()
# config_manager.load_config("bot", "../../telegram_bot/configs/bot.json")
# config_manager.load_config("logger", "../../logs/configs/logger.json")
# config_manager.load_config("parser", "../../tg_parser/configs/parse.json")
# config_manager.load_config("tg_api", "../../tg_parser/configs/tg_api.json")
# config_manager.load_config("usearch_index", "../../text_index/configs/usearch_index.json")
# config_manager.load_config("bm25_index", "../../text_index/configs/bm25_index.json")
# config_manager.load_config("db", "../../db/configs/db_connection.json")
# config_manager.load_config("text", "../../telegram_bot/configs/text_misc.json")
# config_manager.load_config("bot_query", "../../telegram_bot/configs/query.json")

config_manager.load_config("bot", "/home/znai/tgbot/tgbotsb/telegram_bot/configs/bot.json")
config_manager.load_config("logger",  "/home/znai/tgbot/tgbotsb/logs/configs/logger.json")
config_manager.load_config("parser",  "/home/znai/tgbot/tgbotsb/tg_parser/configs/parse.json")
config_manager.load_config("tg_api",  "/home/znai/tgbot/tgbotsb/tg_parser/configs/tg_api.json")
config_manager.load_config("usearch_index",  "/home/znai/tgbot/tgbotsb/text_index/configs/usearch_index.json")
config_manager.load_config("index",  "/home/znai/tgbot/tgbotsb/text_index/configs/index.json")
config_manager.load_config("bm25_index",  "/home/znai/tgbot/tgbotsb/text_index/configs/bm25_index.json")
config_manager.load_config("db",  "/home/znai/tgbot/tgbotsb/db/configs/db_connection.json")
config_manager.load_config("text",  "/home/znai/tgbot/tgbotsb/telegram_bot/configs/text_misc.json")
config_manager.load_config("bot_query",  "/home/znai/tgbot/tgbotsb/telegram_bot/configs/query.json")
config_manager.load_config("cb_ranker",  "/home/znai/tgbot/tgbotsb/cb_ranker/configs/cb_ranker_config.yaml")