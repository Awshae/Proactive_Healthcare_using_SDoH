import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.forecasting import build_lstm_model
import joblib
from utils.config import config
from utils.logger import logger

def train_model(df, features, target):
    X = df[features].values.reshape((df.shape[0], len(features), 1))
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    joblib.dump(model, config.MODEL_PATH)
    logger.info(f"Model trained and saved to {config.MODEL_PATH}")

# Add function for scaling data.
def scale_data(X_train,X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0],-1)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0],-1)).reshape(X_test.shape)
    return X_train_scaled, X_test_scaled
