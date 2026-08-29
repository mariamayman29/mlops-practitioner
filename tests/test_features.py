import pytest
import pandas as pd
from prodml.features import process_data


@pytest.mark.parametrize(
    "duration_mins, expected_rows",
    [
        (15, 1),
        (70, 0),
        (0, 0),
    ],
)
def test_process_data_edge_cases(duration_mins, expected_rows):
    """Test the process_data function with edge cases for trip duration"""
    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": [pd.Timestamp("2026-08-01 10:00:00")],
            "lpep_dropoff_datetime": [
                pd.Timestamp("2026-08-01 10:00:00")
                + pd.Timedelta(minutes=duration_mins)
            ],
            "PULocationID": [43],
            "DOLocationID": [231],
            "trip_distance": [0.0],
        }
    )

    processed = process_data(df)
    assert len(processed) == expected_rows

    if expected_rows == 1:
        assert processed.iloc[0]["PU_DO"] == "43_231"
