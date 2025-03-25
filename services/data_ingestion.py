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
import requests
import os

def fetch_data_from_api(api_url, params=None, headers=None):
    response = requests.get(api_url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        save_raw_data(data, 'api_data.json')
    else:
        print(f"Failed to fetch data: {response.status_code}")

def save_raw_data(data, filename):
    raw_data_dir = 'data/raw/'
    os.makedirs(raw_data_dir, exist_ok=True)
    with open(os.path.join(raw_data_dir, filename), 'w') as f:
        json.dump(data, f)
