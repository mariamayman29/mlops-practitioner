from .config import Config
from .utils.tracking import get_git_commit, get_data_version
from .features import process_data

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Loads the parquet dataset."""
    return pd.read_parquet(Config.data_path)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the dataframe into train/validation sets."""
    return train_test_split(
        df, test_size=Config.test_size, random_state=Config.random_state
    )


def prepare_data():
    """Prepares the data for training"""
    df = load_data()
    logger.info(f"Data loaded successgfully from {Config.data_path}")

    current_commit = get_git_commit()
    data_hash = get_data_version(Config.data_path)

    df = process_data(df)
    logger.info(f"Processed data with shape {df.shape}")

    df_train, df_val = split_data(df)
    logger.info(
        f"Split data into train ({df_train.shape}) and validation ({df_val.shape}) sets"
    )

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    target = "duration"

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    val_dicts = df_val[categorical + numerical].to_dict(orient="records")

    y_train = df_train[target].values
    y_val = df_val[target].values

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    metadata = {
        "current_commit": current_commit,
        "data_hash": data_hash,
        "train_size": len(df_train),
        "val_size": len(df_val),
    }

    return X_train, y_train, X_val, y_val, dv, metadata
