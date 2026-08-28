# Trip Duration Predictor API

A production-ready Machine Learning API that predicts  taxi trip durations. Built with a focus on robust MLOps practices, it transitions a standard scikit-learn model into a highly optimized **ONNX** graph for low-latency inference — served through **FastAPI** with strict Pydantic validation and shipped as a minimal, multi-stage Docker image running as a secure non-root user.

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![ONNX](https://img.shields.io/badge/inference-ONNX-black)
![Docker](https://img.shields.io/badge/container-Docker-2496ED?logo=docker)
![Coverage](https://img.shields.io/badge/coverage-%E2%89%A570%25-brightgreen)

---

## Table of Contents

- [Quickstart: Zero to Prediction in 3 Commands](#quickstart-zero-to-prediction-in-3-commands)
- [MLOps Architecture & Features](#mlops-architecture--features)
- [API Reference](#api-reference)
- [Repository Structure](#repository-structure)
- [Local Development](#local-development)
- [Testing](#testing)
- [Building the Docker Image](#building-the-docker-image)
- [Tech Stack](#tech-stack)


---

## Quickstart: Zero to Prediction in 3 Commands

You do not need to clone this repository or install local Python dependencies to test the system.

**1. Pull the published image from Docker Hub:**
```bash
docker pull mariiamayman/prodml-api:0.1.0
```

**2. Run the container locally:**
```bash
docker run -d -p 8000:8000 mariiamayman/prodml-api:0.1.0
```

**3. Make a prediction using curl:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "PULocationID": 43,
    "DOLocationID": 231,
    "trip_distance": 2.5
  }'
```

Once running, interactive API docs are available at `http://localhost:8000/docs`.

---

## MLOps Architecture & Features

| Feature | Description |
|---|---|
| **High-Performance Serialization** | Uses ONNX instead of Pickle to prevent arbitrary code execution vulnerabilities and drastically reduce inference latency. |
| **Resilient API Design** | Implements FastAPI's `lifespan` context manager to load the model into memory strictly at startup, eliminating per-request loading bottlenecks. |
| **Automated Quality Gates** | Comprehensive Pytest suite using fixtures, mocking and parameterized edge-case testing, enforced by a strict 73% coverage gate. |
| **Optimized Containerization** | Multi-stage Docker build strips heavy C++ build tools from the final image, producing a lightweight, secure footprint. |
| **Structured Logging** | JSON-formatted logs with correlation IDs for request tracing in production. |

---

## API Reference

### `POST /predict`

Predicts trip duration given pickup/dropoff location IDs and trip distance.

**Request body:**
```json
{
  "PULocationID": 43,
  "DOLocationID": 231,
  "trip_distance": 2.5
}
```

**Response:**
```json
{
  "predicted_duration_minutes": 0.0
}
```
### `POST /predict`
Predicts trip duration given pickup/dropoff location IDs and trip distance.

**Request body:**
```json
{
  "PULocationID": 43,
  "DOLocationID": 231,
  "trip_distance": 2.5
}
```
**Response:**
```json
{
  "duration": 14.2
}
```

### `POST /predict/batch`
Accepts a list of trips and returns a list of predicted durations

### `GET /health`
Returns a `200 OK` status,verifying both system liveness and that the ONNX model is successfully loaded into memory
### `GET /metadata`
Returns model metadata, versioning and training run information



---

## Repository Structure

```
.
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfile
├── pyproject.toml
├── src/
│   └── prodml/
│       ├── api/            # FastAPI endpoints and Pydantic schemas
│       ├── config.py
│       ├── data.py         # Data ingestion pipeline
│       ├── export.py       # ONNX serialization and parity tests
│       ├── features.py     # Feature engineering logic
│       ├── logging_conf.py # Structured JSON logger with correlation IDs
│       ├── predict.py      # Inference wrapper
│       └── train.py        # Model training pipeline
└── tests/                  # Pytest suite with mock fixtures
```

---

## Local Development

Clone the repository and install dependencies with your preferred tool (this project uses `pyproject.toml`):

```bash
git clone https://github.com/mariamayman29/mlops-practitioner.git
cd prodml-api
pip install -e ".[dev]"
```

Run the API locally with hot-reload:

```bash
uvicorn prodml.api.main:app --reload --port 8000
```

---

## Testing

Run the full test suite with coverage enforcement:

```bash
pytest --cov=src/prodml --cov-fail-under=70
```

---

## Building the Docker Image

```bash
docker build -f docker/Dockerfile -t prodml-api:local .
```

Or use Docker Compose for local orchestration:

```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## Tech Stack

- **Language:** Python 3.12+
- **API Framework:** FastAPI + Pydantic
- **ML/Serialization:** scikit-learn → ONNX (`skl2onnx`, `onnxruntime`)
- **Testing:** Pytest, pytest-cov, pytest-mock
- **Containerization:** Docker (multi-stage build, non-root user)

---



