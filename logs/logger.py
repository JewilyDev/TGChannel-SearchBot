import logging
from configuration.config import config_manager


logger_cfg = config_manager.get_config("logger")
#logging.basicConfig(level=logging.INFO, filename=logger_cfg['filename'], format=logger_cfg['format'])
logging.basicConfig(level=logging.INFO, filename="/home/znai/tgbot/tgbotsb/logs/main_log.log", format=logger_cfg['format'])

main_logger = logging.getLogger(__name__)
