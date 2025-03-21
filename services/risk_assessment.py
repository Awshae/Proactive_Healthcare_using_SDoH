import joblib
import pandas as pd
from utils.config import config

def generate_forecast(df, features, future_steps):
    model = joblib.load(config.MODEL_PATH)
    last_window = df[features].tail(1).values.reshape((1, len(features), 1))
    forecast = []
    for _ in range(future_steps):
        prediction = model.predict(last_window).flatten()[0]
        forecast.append(prediction)
        last_window = pd.DataFrame(last_window.reshape(1, len(features)), columns = features).shift(-1, axis=1)
        last_window.fillna(prediction, inplace=True)
        last_window = last_window.values.reshape((1, len(features), 1))
    return forecast
