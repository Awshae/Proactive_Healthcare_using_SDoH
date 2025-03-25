import logging
from utils.config import config  # Make sure this path is correct

print(f"Config Loaded: {config.LOG_FILE}")  # Debugging step

def setup_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger(config.LOG_FILE)
