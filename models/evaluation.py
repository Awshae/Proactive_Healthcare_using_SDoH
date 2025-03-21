from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
from utils.config import config

def evaluate_model(df, features, target):
    model = joblib.load(config.MODEL_PATH)
    X = df[features].values.reshape((df.shape[0], len(features), 1))
    y_true = df[target].values
    y_pred = model.predict(X).flatten()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, rmse
