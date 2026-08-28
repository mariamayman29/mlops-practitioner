import pandas as pd
from prodml.train import train_model

def test_train_model_happy_path(monkeypatch, tmp_path):
    """
    Tests the training pipeline by injecting a tiny fake dataset 
    and saving the resulting model to a temporary directory.
    """
    dummy_df = pd.DataFrame({
        "PU_DO": ["43_231", "12_34"],
        "trip_distance": [2.5, 3.0],
        "duration": [14.0, 21.0]
    })
    
 
    monkeypatch.setattr("prodml.train.load_data", lambda: dummy_df)
    monkeypatch.setattr("prodml.train.process_data", lambda df: dummy_df)
    monkeypatch.setattr("prodml.train.split_data", lambda df: (dummy_df, dummy_df))
    
    temp_model_path = tmp_path / "dummy_model.pkl"
    monkeypatch.setattr("prodml.train.Config.model_path", str(temp_model_path))

    train_model()

    assert temp_model_path.exists()