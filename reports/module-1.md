# Module 1 Report

## 1. Project Overview
This project transitions a baseline machine learning model (predicting trip durations) from an experimental Jupyter Notebook into a production-ready, containerized REST API

## 2. Implementation Details
The original notebook was decomposed into a modular architecture:
* `src/prodml/train.py`: Handles data loading, feature extraction (PULocationID, DOLocationID, trip_distance), model training and serialization.
* `src/prodml/predict.py`: Handles model loading (via `onnxruntime`) and inference logic using a lazy-loading singleton pattern.
* `src/prodml/api/`: Contains the FastAPI application, routing and Pydantic schemas for input validation.

## 3. Model Performance & Benchmarks

**Latency Comparison**
As part of the transition to a production-ready API the inference pipeline was upgraded from a standard Scikit-Learn `.pkl` format to ONNX Runtime
* **Pickle Model Latency (500 rows):** 103.69 ms 
* **ONNX Model Latency (500 rows):** 3.36 ms
* **Performance Gain:** ONNX provided an approximate 30.8 x speedup for predictions, alongside improved security by removing arbitrary code execution vulnerabilities associated with pickle files.

**System Metrics**
* **Validation MAE:** 3.85670
* **Validation RMSE:** 5.89132
* **Docker Image Size:** 1.11GB
* **Test Coverage:** 73%

## 4. MLOps Self-Assessment

Based on standard MLOps maturity models, this project has successfully transitioned from **Level 0 (Manual Process)** towards **Level 1 (Automated Pipeline & Continuous Delivery)**

**Current Achievements:**
* **Decomposed Architecture:** Transitioned from Jupyter notebooks to modular version-controlled Python scripts.
* **Containerization:** The inference API is fully encapsulated within a Docker container, ensuring environmental parity between local development and production.
* **Standardized Serving:** Predictions are served via a REST API (FastAPI).
* **Code Quality Gates:** Implemented `pre-commit` hooks (Black, Ruff) to enforce consistent code formatting before code enters the repository.

**Areas for Future Growth (Modules 2+):**
* Currently, the model training and deployment processes are still triggered manually. 
* Future modules will focus on implementing full CI/CD pipelines ( GitHub Actions) to automate testing, building, and deployment, as well as integrating experiment tracking (MLflow) to better monitor training runs and model registries

