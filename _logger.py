import logging


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
