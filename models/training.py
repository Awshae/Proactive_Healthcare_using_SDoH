import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import joblib


import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import config
from forecasting import build_lstm_model 
from utils.logger import logger  


def scale_data(X_train, X_test):
    """Scales the data using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

    # Save scaler for later use in inference
    joblib.dump(scaler, config.SCALER_PATH)
    return X_train_scaled, X_test_scaled

def train_model(df, features, target):
    """Trains an LSTM model using the given dataset"""
    
    # Extract features and target variable
    X = df[features].values.reshape((df.shape[0], len(features), 1))
    y = df[target].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Scale the features
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Build and train LSTM model
    model = build_lstm_model((X_train_scaled.shape[1], 1))
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

    # Save the trained model
    model.save(config.MODEL_PATH)
    logger.info(f"Model trained and saved to {config.MODEL_PATH}")

if __name__ == "__main__":
    # Load data
    df = pd.read_csv(config.PROCESSED_DATA_PATH)

    # Define feature columns and target variable
    feature_columns = [col for col in df.columns if col != "target"]
    target_column = "target"

    # Train the model
    train_model(df, feature_columns, target_column)
