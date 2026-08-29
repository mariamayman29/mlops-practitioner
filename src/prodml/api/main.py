import time
import uuid
import logging
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from prodml.predict import duration_predictor
from prodml.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from prodml.logging_conf import correlation_id_var, setup_logging
from prodml.config import Config

setup_logging()
logger = logging.getLogger(__name__)
predictor = duration_predictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: Loading model into memory")
    predictor.load_model()
    yield
    logger.info("Shutting down")


app = FastAPI(title="Taxi Duration API", lifespan=lifespan)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    req_id = str(uuid.uuid4())
    correlation_id_var.set(req_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422, content={"detail": "Invalid input", "errors": exc.errors()}
    )


@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health_check():
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model unavailable")
    return {"status": "healthy", "model": "loaded"}


@app.get("/metadata")
def metadata():
    return {
        "version": Config.MODEL_VERSION,
        "training_date": "2026-08-27",
        "features": ["PULocationID", "DOLocationID", "trip_distance"],
        "framework": "scikit-learn",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    start = time.time()
    prediction = predictor.predict_trip(payload.model_dump())
    latency = (time.time() - start) * 1000

    return PredictionResponse(
        prediction=prediction,
        model_version=Config.MODEL_VERSION,
        correlation_id=correlation_id_var.get(),
        latency_ms=latency,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest):
    start = time.time()
    trips = [req.model_dump() for req in payload.requests]
    predictions = predictor.predict_trips(trips)
    latency = (time.time() - start) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        model_version=Config.MODEL_VERSION,
        correlation_id=correlation_id_var.get(),
        latency_ms=latency,
    )
