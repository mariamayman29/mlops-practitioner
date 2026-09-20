import subprocess
import os
import hashlib
import numpy as np
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score

def get_git_commit():
    try :
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit_hash
    except subprocess.CalledProcessError:
        return "Unknown"

def get_model_size(file_path:str) ->float :
    try :
        size_in_bytes = os.path.getsize(file_path)
        size_in_megabytes = size_in_bytes /(1024*1024)
        return round(size_in_megabytes ,2)
    except FileNotFoundError :
        return 0.0

def get_data_version(file_path : str) ->str :
    try :
        hasher = hashlib.md5()
        with open (file_path ,'rb') as f :
            for chunk in iter(lambda: f.read(4096),b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError :
        return "Unknown"


def log_environment(project_root: str = ".") -> None:
    candidates = ("pyproject.toml", "poetry.lock", "uv.lock")
    logged_any = False
    for fname in candidates:
        path = os.path.join(project_root, fname)
        if os.path.exists(path):
            mlflow.log_artifact(path, artifact_path="env")
            logged_any = True
    if not logged_any:
        import tempfile
        reqs = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(reqs)
            snapshot_path = f.name
        mlflow.log_artifact(snapshot_path, artifact_path="env")
        os.remove(snapshot_path)

def evaluate_model(y_true, y_pred):
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }
    
    return metrics