from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    trip_distance: float = Field(gt=0, lt=100)
    PULocationID: int = Field(..., description="Pickup Location ID")
    DOLocationID: int = Field(..., description="Dropoff Location ID")

    model_config = {
        "json_schema_extra": {
            "example": {
                "trip_distance": 5.0,
                "PULocationID": 130,
                "DOLocationID": 205
            }
        }
    }

class PredictionResponse(BaseModel):
    prediction : float 
    model_version : str 
    correlation_id : str
    latency_ms : float 

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    correlation_id: str
    latency_ms: float
