from .logging_conf import setup_logging
from .config import Config
from .data import prepare_data
from .utils.tracking import get_model_size, evaluate_model
from .utils.plots import generate_residual_plot, generate_feature_importance_plot

import logging
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import mlflow
import time
import tempfile
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import gc

logger = logging.getLogger(__name__)
mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
mlflow.set_experiment("trip-duration-experiment")

# I only used a small subset of the data to train the models to avoid memory issues on WSL


def train_lr_model() -> None:
    X_train, y_train, X_val, y_val, dv, metadata = prepare_data()
    X_train, y_train = X_train[:500], y_train[:500]
    X_val, y_val = X_val[:100], y_val[:100]

    with mlflow.start_run(run_name="linear_regression_baseline"):
        mlflow.set_tag("git_commit", metadata["current_commit"])
        mlflow.set_tag("data_version", metadata["data_hash"])
        mlflow.set_tag("author", "mariam")
        mlflow.set_tag("framework", "sklearn")

        logger.info("training the linear regression baseline model")

        lr = LinearRegression()
        start_time = time.time()
        lr.fit(X_train, y_train)
        train_duration = time.time() - start_time

        y_pred = lr.predict(X_val)
        metrics = evaluate_model(y_val, y_pred)
        metrics["train_duration"] = train_duration

        residual_plot_path = generate_residual_plot(
            y_val, y_pred, filename="residual_plot.png"
        )
        mlflow.log_artifact(residual_plot_path, artifact_path="plots")
        os.remove(residual_plot_path)

        lr_importance = dict(zip(dv.get_feature_names_out(), np.abs(lr.coef_)))
        importance_path = generate_feature_importance_plot(
            lr_importance, filename="lr_feature_importance.png"
        )
        mlflow.log_artifact(importance_path, artifact_path="plots")
        os.remove(importance_path)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            pickle.dump((dv, lr), temp_file)
            temp_file.flush()
            metrics["model_size_MB"] = get_model_size(temp_file.name)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr, artifact_path="model")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_dv:
            pickle.dump(dv, temp_dv)
            temp_dv.flush()
            mlflow.log_artifact(temp_dv.name, artifact_path="preprocessor")


def train_xgb_sweep(num_trials=10) -> None:
    X_train, y_train, X_val, y_val, dv, metadata = prepare_data()
    X_train, y_train = X_train[:500], y_train[:500]
    X_val, y_val = X_val[:100], y_val[:100]

    logger.info(f"Training XGBoost hyperparameter sweep with {num_trials} trials")

    train_data = xgb.DMatrix(X_train, label=y_train)
    val_data = xgb.DMatrix(X_val, label=y_val)

    def objective(params):
        with mlflow.start_run(run_name="xgboost_trial", nested=True):
            mlflow.set_tag("git_commit", metadata["current_commit"])
            mlflow.set_tag("author", "mariam")
            mlflow.set_tag("framework", "xgboost")
            mlflow.set_tag("data_version", metadata["data_hash"])

            mlflow.log_param("split_seed", Config.random_state)
            mlflow.log_param("data_version", metadata["data_hash"])
            mlflow.log_params(params)

            start_time = time.time()
            booster = xgb.train(
                params=params,
                dtrain=train_data,
                num_boost_round=8,
                evals=[(val_data, "validation")],
                early_stopping_rounds=2,
                verbose_eval=False,
            )
            train_duration = time.time() - start_time

            y_pred = booster.predict(val_data)
            metrics = evaluate_model(y_val, y_pred)
            metrics["train_duration"] = train_duration

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_model:
                pickle.dump((dv, booster), temp_model)
                temp_model.flush()
                metrics["model_size_MB"] = get_model_size(temp_model.name)

            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(booster, artifact_path="model")

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_dv:
                pickle.dump(dv, temp_dv)
                temp_dv.flush()
                mlflow.log_artifact(temp_dv.name, artifact_path="preprocessor")

            residual_path = generate_residual_plot(
                y_val, y_pred, filename="xgb_residuals.png"
            )
            mlflow.log_artifact(residual_path, artifact_path="plots")
            os.remove(residual_path)

            importance = booster.get_score(importance_type="gain")
            importance_path = generate_feature_importance_plot(
                importance, filename="xgb_feature_importance.png"
            )
            mlflow.log_artifact(importance_path, artifact_path="plots")
            os.remove(importance_path)

            plt.close("all")
            gc.collect()

            return {"loss": metrics.get("rmse", 9999), "status": STATUS_OK}

    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 3, 5, 1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
        "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
        "objective": "reg:squarederror",
        "seed": 42,
    }

    with mlflow.start_run(run_name="xgboost_sweep_parent_2"):
        mlflow.log_param("num_trials", num_trials)
        best_result = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=num_trials,
            trials=Trials(),
        )
        logger.info(f"Best hyperparameters found: {best_result}")


def train_pytorch_mlp() -> None:
    X_train_full, y_train_full, X_val_full, y_val_full, dv, metadata = prepare_data()

    X_train = X_train_full[:500].copy()
    y_train = y_train_full[:500].copy()
    X_val = X_val_full[:100].copy()
    y_val = y_val_full[:100].copy()

    del X_train_full, y_train_full, X_val_full, y_val_full
    gc.collect()

    logger.info("Training PyTorch MLP model")

    X_train_tensor = torch.FloatTensor(X_train.toarray())
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val.toarray())

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    input_dim = X_train_tensor.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 4

    with mlflow.start_run(run_name="pytorch_mlp_baseline"):
        mlflow.set_tag("git_commit", metadata["current_commit"])
        mlflow.set_tag("data_version", metadata["data_hash"])
        mlflow.set_tag("author", "mariam")
        mlflow.set_tag("framework", "pytorch")

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("batch_size", 128)
        mlflow.log_param("split_seed", Config.random_state)
        mlflow.log_param("data_version", metadata["data_hash"])

        start_time = time.time()
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
        train_duration = time.time() - start_time

        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(X_val_tensor)
            y_pred = y_pred_tensor.numpy().flatten()

        metrics = evaluate_model(y_val, y_pred)
        metrics["train_duration"] = train_duration

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_model:
            torch.save(model.state_dict(), temp_model.name)
            metrics["model_size_MB"] = get_model_size(temp_model.name)
            os.remove(temp_model.name)

        mlflow.log_metrics(metrics)

        example_input = X_val_tensor[:1].numpy()
        mlflow.pytorch.log_model(
            model, artifact_path="model", input_example=example_input
        )

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_dv:
            pickle.dump(dv, temp_dv)
            temp_dv.flush()
            mlflow.log_artifact(temp_dv.name, artifact_path="preprocessor")

        residual_path = generate_residual_plot(
            y_val, y_pred, filename="mlp_residuals.png"
        )
        mlflow.log_artifact(residual_path, artifact_path="plots")
        os.remove(residual_path)

        first_layer_weights = model[0].weight.detach().numpy()
        mlp_importance = dict(
            zip(dv.get_feature_names_out(), np.abs(first_layer_weights).sum(axis=0))
        )
        importance_path = generate_feature_importance_plot(
            mlp_importance, filename="mlp_feature_importance.png"
        )
        mlflow.log_artifact(importance_path, artifact_path="plots")
        os.remove(importance_path)


if __name__ == "__main__":
    setup_logging()
    train_lr_model()
    train_xgb_sweep(num_trials=10)
    train_pytorch_mlp()
