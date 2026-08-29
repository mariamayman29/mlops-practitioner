def test_predict_one_deterministic(trained_model, sample_features):
    """Test that the model's predict_one method is deterministic for the same input"""
    pred1 = trained_model.predict_trip(sample_features)
    pred2 = trained_model.predict_trip(sample_features)

    assert isinstance(pred1, float)
    assert 0 < pred1 < 120
    assert pred1 == pred2
