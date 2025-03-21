import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_URL_UNEMPLOYMENT = os.getenv("API_URL_UNEMPLOYMENT")
    API_KEY_UNEMPLOYMENT = os.getenv("API_KEY_UNEMPLOYMENT")
    DATABASE_URL = os.getenv("DATABASE_URL")
    SECRET_KEY = os.getenv("SECRET_KEY")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/trained_model.pkl")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")
    DATA_DIR_RAW = os.getenv("DATA_DIR_RAW", "data/raw")
    DATA_DIR_PROCESSED = os.getenv("DATA_DIR_PROCESSED", "data/processed")

config = Config()
