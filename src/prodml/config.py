from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    data_path:str = "data/green_tripdata_2023-01.parquet"
    model_path:str = "models/baseline.pkl"
    test_size:float = 0.2
    random_state:int = 42


    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

Config = Settings()