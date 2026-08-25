import pandas as pd

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the input DataFrame by converting datetime columns to datetime objects,
    calculating the duration of trips, and filtering out trips with durations outside
    the range of 1 to 60 minutes.
    """

    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)

    return df