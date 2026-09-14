from numpy import tri
from .logging_conf import setup_logging
from .config import Config
import pickle
import time 
import logging 
from functools import wraps


logger = logging.getLogger(__name__)

def timed(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result= func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

class duration_predictor:
    """Class for predicting trip durations using a pre-trained model"""
    def __init__(self):
        self.model = None
        self.dv = None
        self.is_loaded = False

    def load_model(self):

        if not self.is_loaded:
            logger.info(f"Loading model from {Config.model_path}")
            with open(Config.model_path, 'rb') as f:
                self.dv, self.model = pickle.load(f)
            self.is_loaded = True
            logger.info("Model loaded successfully")
        else:
            logger.info("Model is already loaded")

    @timed
    def predict_trip(self,trip_info: dict) -> float:
        if not self.is_loaded:
            self.load_model()

        trip_info['PU_DO'] = str(trip_info['PULocationID']) + '_' + str(trip_info['DOLocationID'])
        X = self.dv.transform([trip_info])

        predicted_duration = self.model.predict(X)[0]
        logger.info(f"Predicted duration for trip {trip_info}: {predicted_duration:.2f} minutes")
        return predicted_duration

    @timed
    def predict_trips(self, trips_info: list[dict]) -> list[float]:
        if not self.is_loaded:
            self.load_model()

        for trip in trips_info:
            trip['PU_DO'] = str(trip['PULocationID']) + '_' + str(trip['DOLocationID'])
        
        X = self.dv.transform(trips_info)
        predicted_durations = self.model.predict(X)
        logger.info(f"Predicted durations for trips: {predicted_durations.tolist()}")
        return predicted_durations.tolist()

if __name__ == "__main__":
    setup_logging()


   