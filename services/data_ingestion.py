import requests
import pandas as pd
from utils.config import config
from utils.logger import logger

def fetch_unemployment_data():
    try:
        headers = {"Authorization": f"Bearer {config.API_KEY_UNEMPLOYMENT}"}
        response = requests.get(config.API_URL_UNEMPLOYMENT, headers=headers)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching unemployment data: {e}")
        return None

# Add functions for other data sources (evictions, school attendance, etc.)
