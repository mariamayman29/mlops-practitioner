import pytest
from fastapi.testclient import TestClient
from prodml.api.main import app
from prodml.predict import duration_predictor


@pytest.fixture(scope="session")
def trained_model():
    predictor = duration_predictor()
    predictor.load_model()
    return predictor


@pytest.fixture
def sample_features():
    return {"PULocationID": 43, "DOLocationID": 231, "trip_distance": 2.5}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
