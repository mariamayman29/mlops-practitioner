from .logging_conf import setup_logging
from .config import Config
import pickle
import time
import mlflow
from mlflow.tracking import MlflowClient
import logging
from functools import wraps
import os


logger = logging.getLogger(__name__)
MLFLOW_TRACKING_URI = Config.MLFLOW_TRACKING_URI


def timed(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds"
        )
        return result

    return wrapper


class duration_predictor:
    """Class for predicting trip durations using a pre-trained model"""

    def __init__(self, model_name="ride-duration-predictor", stage="Production"):
        self.model = None
        self.dv = None
        self.is_loaded = False
        self.model_name = model_name
        self.stage = stage

    def load_model(self):
        if not self.is_loaded:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()

            logger.info(f"Loading {self.stage} model '{self.model_name}' from MLflow")
            model_uri = f"models:/{self.model_name}/{self.stage}"
            self.model = mlflow.pyfunc.load_model(model_uri)

            latest_versions = client.get_latest_versions(
                name=self.model_name, stages=[self.stage]
            )
            if not latest_versions:
                raise ValueError(
                    f"No model found for '{self.model_name}' in stage '{self.stage}'"
                )
            production_run_id = latest_versions[0].run_id

            logger.info(f"Downloading preprocessor for run_id: {production_run_id}")
            preprocessor_dir = client.download_artifacts(
                run_id=production_run_id, path="preprocessor"
            )

            for file in os.listdir(preprocessor_dir):
                if file.endswith(".pkl"):
                    with open(os.path.join(preprocessor_dir, file), "rb") as f:
                        self.dv = pickle.load(f)
                    break

            if not self.dv:
                raise FileNotFoundError(
                    "DictVectorizer (.pkl) not found in the MLflow artifact!"
                )

            self.is_loaded = True
            logger.info("Model and DictVectorizer loaded successfully from MLflow")
        else:
            logger.info("Model is already loaded")

    @timed
    def predict_trip(self, trip_info: dict) -> float:
        if not self.is_loaded:
            self.load_model()

        trip_info["PU_DO"] = (
            str(trip_info["PULocationID"]) + "_" + str(trip_info["DOLocationID"])
        )
        X = self.dv.transform([trip_info])

        predicted_duration = self.model.predict(X)[0]
        logger.info(
            f"Predicted duration for trip {trip_info}: {predicted_duration:.2f} minutes"
        )
        return predicted_duration

    @timed
    def predict_trips(self, trips_info: list[dict]) -> list[float]:
        if not self.is_loaded:
            self.load_model()

        for trip in trips_info:
            trip["PU_DO"] = str(trip["PULocationID"]) + "_" + str(trip["DOLocationID"])

        X = self.dv.transform(trips_info)
        predicted_durations = self.model.predict(X)
        logger.info(f"Predicted durations for trips: {predicted_durations.tolist()}")
        return predicted_durations.tolist()


if __name__ == "__main__":
    setup_logging()
