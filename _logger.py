import logging
from colorama import Fore, Style


class ColorFormatter(logging.Formatter):
    COLOR_MAP = {
        logging.INFO: Fore.BLUE,  # Info messages in blue
        logging.WARNING: Fore.YELLOW,  # Warning messages in yellow
        logging.ERROR: Fore.RED,  # Error messages in red
        logging.DEBUG: Fore.GREEN,  # Optional: Debug messages in green
        logging.CRITICAL: Fore.MAGENTA,  # Optional: Critical messages in magenta
    }
    
    def format(self, record):
        color = self.COLOR_MAP.get(record.levelno, Fore.WHITE)  # Default to white
        log_msg = super().format(record)
        return f"{color}{log_msg}{Style.RESET_ALL}"


class ProjectLogger:
    def __init__(self, class_name:str) -> None:
        self.class_name = class_name

    def create_logger(self):
        logger = logging.getLogger(name=self.class_name)
        logger.setLevel(level=logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.info(msg='Logger created successfully!')
        return logger
