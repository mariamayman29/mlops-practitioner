import mlflow
from mlflow.tracking import MlflowClient
import logging 
from .config import Config

mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
client = MlflowClient(Config.MLFLOW_TRACKING_URI)
logger = logging.getLogger(__name__)

# I had some issues with mlflow UI not showing the registered models, so I had to register the best and worst models manually

best_run_id = "ecf29248093d4ebfb5e6d370993a53fd"
mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model", 
    name="ride-duration-predictor",
)


worst_run_id = "8f6912e4a23f43f9b7a49e626a7b3a39" 
mlflow.register_model(
    model_uri=f"runs:/{worst_run_id}/model", 
    name="ride-duration-predictor"
)

client.transition_model_version_stage(
    name="ride-duration-predictor",
    version=2,
    stage="Production"
)


def promote_if_better(candidate_run_id: str, model_name: str = "ride-duration-predictor", metric: str = "mae"):
    """Compares a candidate run to the current Production model and promotes it if it performs better"""

    logger.info(f"Evaluating candidate run: {candidate_run_id}")
    
    candidate_run = mlflow.get_run(candidate_run_id)
    candidate_metric = candidate_run.data.metrics.get(metric)
    if candidate_metric is None:
        raise ValueError(f"Candidate run does not have the metric '{metric}' logged.")
        
    logger.info(f"Candidate {metric}: {candidate_metric:.4f}")

    try:
        latest_versions = client.get_latest_versions(name=model_name, stages=["Production"])
        if not latest_versions:
            raise ValueError("No Production model found.")
            
        prod_run_id = latest_versions[0].run_id
        prod_run = mlflow.get_run(prod_run_id)
        prod_metric = prod_run.data.metrics.get(metric)
        logger.info(f"Current Production {metric}: {prod_metric:.4f}")
        
    except Exception as e:
        logger.warning(f"Notice: {e} Promoting candidate automatically.")
        prod_metric = float('inf') 
    if candidate_metric < prod_metric:
        logger.info(f"Candidate beat Production! ({candidate_metric:.4f} < {prod_metric:.4f}). Registering...")
        
        new_version = mlflow.register_model(
            model_uri=f"runs:/{candidate_run_id}/model",
            name=model_name
        )
        
        client.transition_model_version_stage(
            name=model_name,
            version=new_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"Successfully promoted Version {new_version.version} to Production!")
    else:
        logger.info(f"Candidate did not beat Production ({candidate_metric:.4f} >= {prod_metric:.4f}). No action taken.")


if __name__ == "__main__":
    best_xgb_run_id = "c824d448ef054f6dbb002f2339621843"
    promote_if_better(candidate_run_id=best_xgb_run_id, metric="mae")