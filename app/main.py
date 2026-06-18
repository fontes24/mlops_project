import io
import logging
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import load_model

logger = logging.getLogger("app.main")


class ModelService:
    def __init__(self) -> None:
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all artifacts from the local project folder."""
        logger.info("Loading artifacts from local project folder")

        # Define base paths
        artifacts_dir = "artifacts"
        models_dir = "models"

        # Define paths to the preprocessing artifacts
        features_imputer_path = os.path.join(
            artifacts_dir, "[features]_mean_imputer.joblib"
        )
        features_scaler_path = os.path.join(artifacts_dir, "[features]_scaler.joblib")
        target_encoder_path = os.path.join(
            artifacts_dir, "[target]_one_hot_encoder.joblib"
        )
        # Define path to the model file
        model_path = os.path.join(models_dir, "model.keras")

        # Load all required artifacts
        self.features_imputer = joblib.load(features_imputer_path)
        self.features_scaler = joblib.load(features_scaler_path)
        self.target_encoder = joblib.load(target_encoder_path)
        self.model = load_model(model_path)

        logger.info("Successfully loaded all artifacts")

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Make predictions using the full pipeline.

        Args:
            features: DataFrame containing the input features

        Returns:
            Series containing the predictions
        """
        # Apply transformations in sequence
        X_imputed = self.features_imputer.transform(features)
        X_scaled = self.features_scaler.transform(X_imputed)

        # Get model predictions
        y_pred = self.model.predict(X_scaled)

        # Decode predictions
        y_decoded = self.target_encoder.inverse_transform(y_pred)

        return pd.DataFrame({"Prediction": y_decoded.ravel()}, index=features.index)


def create_routes(app: Flask) -> None:
    """Create all routes for the application."""

    @app.route("/")
    def index() -> str:
        """Serve the HTML upload interface."""
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload() -> str:
        """Handle CSV file upload, validate features, and return predictions."""
        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return render_template("index.html", error="Please upload a CSV file")

        try:
            # Read CSV content
            content = file.read().decode("utf-8")
            features = pd.read_csv(io.StringIO(content))

            # Validate column names against breast cancer dataset
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

            # Make predictions
            predictions = app.model_service.predict(features)

            # Format predictions for display
            result = predictions.to_string()

            return render_template("index.html", predictions=result)

        except Exception as e:
            logger.error(
                f"Error processing file: {e}", exc_info=True
            )  # Added exc_info for better logging
            return render_template(
                "index.html",
                error=f"Error processing file: {str(e)}",  # Ensure e is string
            )


# Create and configure Flask app at module level
app = Flask(__name__)
app.model_service = ModelService()
create_routes(app)
logger.info("Application initialized with model service and routes")


def main() -> None:
    """Run the Flask development server."""
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()
