from .config import Config
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data() -> pd.DataFrame:
    """Loads the parquet dataset."""
    return pd.read_parquet(Config.data_path)

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the dataframe into train/validation sets."""
    return train_test_split(
        df, 
        test_size=Config.test_size, 
        random_state=Config.random_state
    )