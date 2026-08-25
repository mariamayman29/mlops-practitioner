from .config import Config
import pickle

def load_model() -> tuple:
    """Loads the trained model from disk."""
    with open(Config.model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def predict(features: dict) -> float:
    """Makes a prediction using the trained model."""
    dv, model = load_model()
    X = dv.transform([features])
    prediction = model.predict(X)
    return prediction[0]

if __name__ == "__main__":
    sample_trip = {
        "PULocationID": 43,
        "DOLocationID": 231,
        "trip_distance": 2.5
    }
    
    predicted_duration = predict(sample_trip)
    print(f"Predicted trip duration: {predicted_duration:.2f} minutes")