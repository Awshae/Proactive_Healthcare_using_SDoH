import pandas as pd
from utils.logger import logger

def create_lag_features(df, columns, lags):
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df

def create_rolling_window_features(df, columns, windows):
    for col in columns:
        for window in windows:
            df[f"{col}_rolling_{window}_mean"] = df[col].rolling(window=window).mean()
    return df

def create_time_features(df):
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    return df
