from .config import Config
from .data import load_data, split_data
from .features import process_data, process_data

import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error



def train_model() -> None:
    """Trains the baseline model and saves it to disk"""

    df= load_data()
    
    df = process_data(df)

    df_train, df_val = split_data(df)

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    target = 'duration'
    
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    
    y_train = df_train[target].values
    y_val = df_val[target].values

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"Validation MAE: {mae:.5f}")
    print(f"Validation RMSE: {rmse:.5f}")
    
    print(f"Saving model to {Config.model_path}")
    with open(Config.model_path, 'wb') as f:
        pickle.dump((dv, lr), f)
        

if __name__ == "__main__":
    train_model()