import io
import logging
import os

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from sklearn.datasets import load_breast_cancer
from mlflow.tracking import MlflowClient

load_dotenv()

logger = logging.getLogger("app.main")

DEFAULT_MODEL_URI = "models:/model/@production"
DEFAULT_MODEL_NAME = "model"


def _parse_dagshub_repo_from_tracking_uri(uri: str):
    """Return (owner, repo) parsed from a DagsHub tracking URI, or (None, None)."""
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


def _configure_mlflow() -> None:
    """Configure MLflow tracking URI and DagsHub authentication.

    DagsHub rejects anonymous Model Registry requests with 403. The official
    ``dagshub`` Python library injects the correct Basic auth header when we
    call ``dagshub.init(...)`` and ``dagshub.auth.add_token(...)``. The
    owner/repo are derived from ``MLFLOW_TRACKING_URI`` when not set as
    standalone env vars.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to {tracking_uri}")

    token = os.getenv("DAGSHUB_USER_TOKEN")
    owner = os.getenv("DAGSHUB_REPO_OWNER")
    name = os.getenv("DAGSHUB_REPO_NAME")
    if not owner or not name:
        owner, name = _parse_dagshub_repo_from_tracking_uri(tracking_uri or "")

    if token and owner and name:
        import dagshub

        dagshub.auth.add_app_token(token)
        dagshub.init(repo_owner=owner, repo_name=name, mlflow=True)
        logger.info(f"DagsHub auth configured for {owner}/{name}")
    elif not token:
        logger.warning(
            "DAGSHUB_USER_TOKEN is not set; authenticated MLflow calls will "
            "fail with 403."
        )


_configure_mlflow()


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.features_imputer = None
        self.features_scaler = None
        self.target_encoder = None
        self.model_uri = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)
        self.model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        self._load_artifacts()

    def _resolve_version(self) -> int:
        """Resolve a model URI to a concrete version number.

        DagsHub's MLflow proxy still routes ``models:/<name>/@<alias>`` through
        the legacy ``get_latest_versions(stage=...)`` endpoint, which rejects
        alias-style references with ``INVALID_PARAMETER_VALUE``. We resolve the
        alias to a numeric version client-side first, then load by version.
        """
        client = MlflowClient()
        if "@" in self.model_uri:
            try:
                alias = self.model_uri.split("@", 1)[1]
                version = client.get_model_version_by_alias(
                    self.model_name, alias
                ).version
                logger.info(
                    f"Resolved alias '@{alias}' of '{self.model_name}' to "
                    f"version {version}"
                )
                return int(version)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Could not resolve alias in {self.model_uri}: {exc}. "
                    "Falling back to latest version."
                )
        # Fallback: pick the highest version for the model.
        versions = client.search_model_versions(f"name='{self.model_name}'")
        if not versions:
            raise mlflow.exceptions.MlflowException(
                f"No versions found for model '{self.model_name}'"
            )
        return max(int(v.version) for v in versions)

    def _load_artifacts(self):
        """Load the registered model from MLflow Model Registry and related artifacts from its run.

        Failures are logged and the service is left in a degraded state; endpoints
        check ``self.is_ready()`` and return 503 when the model is not available.
        """
        logger.info(f"Loading registered model from MLflow Model Registry at {self.model_uri}")
        try:
            version = self._resolve_version()
            versioned_uri = f"models:/{self.model_name}/{version}"
            self.model = mlflow.keras.load_model(versioned_uri)

            client = MlflowClient()
            mv = client.get_model_version(self.model_name, str(version))
            run_id = mv.run_id

            logger.info(f"Loading artifacts from run {run_id}")
            artifacts_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")

            imputer_path = os.path.join(artifacts_dir, "[features]_mean_imputer.joblib")
            self.features_imputer = joblib.load(imputer_path)
            scaler_path = os.path.join(artifacts_dir, "[features]_scaler.joblib")
            self.features_scaler = joblib.load(scaler_path)
            encoder_path = os.path.join(artifacts_dir, "[target]_one_hot_encoder.joblib")
            self.target_encoder = joblib.load(encoder_path)

            logger.info("Successfully loaded model and related artifacts")
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Failed to load model/artifacts from {self.model_uri}: {exc}",
                exc_info=True,
            )
            self.model = None
            self.features_imputer = None
            self.features_scaler = None
            self.target_encoder = None

    def is_ready(self) -> bool:
        return (
            self.model is not None
            and self.features_imputer is not None
            and self.features_scaler is not None
            and self.target_encoder is not None
        )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the full pipeline.

        Args:
            features: DataFrame containing the input features

        Returns:
            DataFrame containing the predictions
        """
        X_imputed = self.features_imputer.transform(features)
        X_scaled = self.features_scaler.transform(X_imputed)

        y_pred = self.model.predict(X_scaled)

        y_decoded = self.target_encoder.inverse_transform(y_pred)

        return pd.DataFrame({"Prediction": y_decoded.ravel()}, index=features.index)


def create_routes(app: Flask) -> None:
    """Create all routes for the application."""

    @app.route("/")
    def index() -> str:
        """Serve the HTML upload interface."""
        return render_template("index.html")

    @app.route("/health")
    def health():
        """Liveness/readiness probe. Returns 200 with ready=true, or 503 when degraded."""
        ready = app.model_service.is_ready()
        payload = {
            "ready": ready,
            "model_uri": app.model_service.model_uri,
        }
        return jsonify(payload), (200 if ready else 503)

    @app.route("/upload", methods=["POST"])
    def upload() -> str:
        """Handle CSV file upload, validate features, and return predictions."""
        if not app.model_service.is_ready():
            return render_template(
                "index.html",
                error="Model is not available. Please check the service logs and try again later.",
            ), 503

        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return render_template("index.html", error="Please upload a CSV file")

        try:
            content = file.read().decode("utf-8")
            features = pd.read_csv(io.StringIO(content))

            expected_features = load_breast_cancer().feature_names
            missing_cols = [
                col for col in expected_features if col not in features.columns
            ]
            if missing_cols:
                return render_template(
                    "index.html",
                    error=f"Missing required columns: {', '.join(missing_cols)}",
                )
            features = features[expected_features]

            predictions = app.model_service.predict(features)
            result = predictions.to_string()

            return render_template("index.html", predictions=result)

        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            return render_template(
                "index.html",
                error=f"Error processing file: {str(e)}",
            )


app = Flask(__name__)
app.model_service = ModelService()
create_routes(app)
logger.info(
    f"Application initialized (model ready: {app.model_service.is_ready()})"
)


def main() -> None:
    """Run the Flask development server."""
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()
