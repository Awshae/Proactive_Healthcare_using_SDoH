import tensorflow as tf
from tensorflow import keras
from utils.logger import logger

def build_lstm_model(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
