import logging
from .logging_conf import setup_logging
from .config import Config
from .data import load_data, split_data
from .features import process_data, process_data

import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

def train_model() -> None:
    """Trains the baseline model and saves it to disk"""

    df= load_data()
    logger.info(f"Loaded data from {Config.data_path} with shape {df.shape}")

    df = process_data(df)
    logger.info(f"Processed data with shape {df.shape}")

    df_train, df_val = split_data(df)
    logger.info(f"Split data into train ({df_train.shape}) and validation ({df_val.shape}) sets")

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    target = 'duration'
    
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    
    y_train = df_train[target].values
    y_val = df_val[target].values

    logger.info("Training the model...")

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    logger.info(f"Validation MAE: {mae:.5f}")
    logger.info(f"Validation RMSE: {rmse:.5f}")
    
    logger.info(f"Saving model to {Config.model_path}")
    with open(Config.model_path, 'wb') as f:
        pickle.dump((dv, lr), f)
        

if __name__ == "__main__":
    setup_logging()
    train_model()