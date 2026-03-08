import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv; load_dotenv()

LOG_NAME = "wine-quality-classifier"
LOG_DIR = "logs"
LOG_FILENAME = os.path.join(LOG_DIR, "app.log")

ENV = os.getenv("ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if ENV == "dev" else "INFO")

def get_logger(name: str = LOG_NAME):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level=LOG_LEVEL)
        
        # 1. console logger
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.stream.reconfigure(encoding="utf-8", errors="replace")
        console_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s'))
        logger.addHandler(console_handler)

        if ENV == "prod":
            # 2. setup file handler
            os.makedirs(name=LOG_DIR, exist_ok=True)
            file_handler = RotatingFileHandler(
                LOG_FILENAME, 
                maxBytes=5242880, 
                backupCount=3
            )
            file_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s'))
            logger.addHandler(file_handler)
   
    return logger

log = get_logger()