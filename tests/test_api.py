def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_happy_path(client, sample_features, monkeypatch):
    monkeypatch.setattr("prodml.api.main.predictor.predict_trip", lambda x: 14.2)
    
    response = client.post("/predict", json=sample_features)
    assert response.status_code == 200
    assert response.json()["prediction"] == 14.2

def test_predict_invalid_payload(client):
    bad_payload = {"PULocationID": 43, "DOLocationID": 231, "trip_distance": -5}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422