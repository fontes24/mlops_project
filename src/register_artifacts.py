import logging
import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
import pandas as pd

load_dotenv()

logger = logging.getLogger("src.register_artifacts")

REGISTERED_MODEL_NAME = os.getenv("MODEL_NAME", "model")
PRODUCTION_ALIAS = os.getenv("PRODUCTION_ALIAS", "production")
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")


def _parse_dagshub_repo_from_tracking_uri(uri: str):
    """Return (owner, repo) parsed from a DagsHub tracking URI, or (None, None).

    Examples:
        https://dagshub.com/fontes24/mlops_project.mlflow
            -> ("fontes24", "mlops_project")
        https://dagshub.com/fontes24/mlops_project.mlflow/
            -> ("fontes24", "mlops_project")
    """
    if not uri:
        return None, None
    try:
        from urllib.parse import urlparse

        parts = [p for p in urlparse(uri).path.split("/") if p]
    except Exception:  # noqa: BLE001
        return None, None
    if len(parts) < 2:
        return None, None
    owner, repo = parts[0], parts[1]
    if repo.endswith(".mlflow"):
        repo = repo[: -len(".mlflow")]
    return owner, repo


def _configure_mlflow_auth() -> None:
    """Configure MLflow tracking URI and DagsHub authentication.

    DagsHub's Model Registry API rejects anonymous requests with 403
    ("Only signed in user is allowed to call APIs."), so the request must
    carry Basic auth. The official ``dagshub`` Python library wires this up
    for us when we call ``dagshub.init(...)`` and ``dagshub.auth.add_token``.
    Owner/repo are derived from ``MLFLOW_TRACKING_URI`` when not set as
    standalone env vars.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to {tracking_uri}")

    owner = os.getenv("DAGSHUB_REPO_OWNER")
    name = os.getenv("DAGSHUB_REPO_NAME")
    if not owner or not name:
        owner, name = _parse_dagshub_repo_from_tracking_uri(tracking_uri or "")

    if DAGSHUB_USER_TOKEN and owner and name:
        import dagshub  # lazy import for environments without dagshub

        dagshub.auth.add_app_token(DAGSHUB_USER_TOKEN)
        dagshub.init(repo_owner=owner, repo_name=name, mlflow=True)
        logger.info(f"DagsHub auth configured for {owner}/{name}")
    elif not DAGSHUB_USER_TOKEN:
        logger.warning(
            "DAGSHUB_USER_TOKEN is not set; authenticated MLflow calls will "
            "fail with 403."
        )


_configure_mlflow_auth()

client = MlflowClient()


def get_best_run(experiment_id: str, parent_run_id: str) -> pd.Series:
    """Get the best child run based on test accuracy for a given parent run.
    
    Args:
        client: MLflow client instance
        parent_run_id: ID of the parent run
        
    Returns:
        The best run as a pandas Series
    """
    # Get all child runs for the parent
    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        order_by=["metrics.test_accuracy DESC"],
        max_results=1000
    )       
    # Return the run with highest test accuracy
    return child_runs[0]

def register_model() -> None:
    """Register the model that was logged during training."""

    logger.info("Registering model from latest MLflow run")

    # Get the experiment ID for the 'ml_classification' experiment
    experiment_id = client.get_experiment_by_name("ml_classification").experiment_id

    # Get the latest run from the experiment
    latest_run = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )[0]
    
    # Check if the latest run has a parent run
    run_id = latest_run.info.run_id
    parent_run_id = latest_run.data.tags.get('mlflow.parentRunId')
    
    if parent_run_id:
        logger.info(f"Latest run has parent run ID: {parent_run_id}")
        best_run = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            order_by=["metrics.test_accuracy DESC"],
            max_results=1
        )[0]
        run_id = best_run.info.run_id
        logger.info(f"Using best run {run_id} with test_accuracy: {best_run.data.metrics['test_accuracy']}")

    # Register the model from the run
    logger.info(f"Registering model '{REGISTERED_MODEL_NAME}'")
    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
        logger.info(f"Created registered model '{REGISTERED_MODEL_NAME}'")
    except mlflow.exceptions.RestException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
        logger.debug(f"Registered model '{REGISTERED_MODEL_NAME}' already exists")

    model_uri = f"runs:/{run_id}/model"
    # ``create_model_version`` blocks up to 300s waiting for READY on some
    # registries (notably DagsHub). We don't need to wait here because
    # ``set_registered_model_alias`` accepts PENDING_REGISTRATION versions and
    # the runtime (app/main.py) loads artifacts directly via run_id.
    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=model_uri,
        run_id=run_id,
    )
    version = model_version.version
    logger.info(
        f"Registered model version {version} for '{REGISTERED_MODEL_NAME}' "
        f"(status: {model_version.status})"
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=PRODUCTION_ALIAS,
        version=version,
    )
    logger.info(
        f"Promoted version {version} of '{REGISTERED_MODEL_NAME}' to alias "
        f"'{PRODUCTION_ALIAS}'"
    )


def main() -> None:
    """Main function to orchestrate the model registration process."""
    register_model()
    logger.info("Model registration completed")


if __name__ == "__main__":
    main()
