import logging
import os
from .config import Config
from datetime import datetime


def setup_logging(filename):
    filename = filename + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    file_handler = logging.FileHandler(os.path.join(Config.LOG_DIR, filename), mode="w")
    file_handler.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.basicConfig(level= logging.DEBUG, format = '%(message)s', handlers = [file_handler, console])
    logger = logging.getLogger(filename)
    logger.debug(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
    return logger
