import logging
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("src.model_evaluation.evaluate_model")


def load_model() -> tf.keras.Model:
    """Load the trained Keras model from disk.

    Returns:
        tf.keras.Model: Loaded Keras model.
    """
    model_path = "models/model.keras"
    model = tf.keras.models.load_model(model_path)
    return model


def load_encoder() -> LabelEncoder:
    """Load the label encoder from disk.

    Returns:
        LabelEncoder: Loaded label encoder.
    """
    encoder_path = "artifacts/[target]_one_hot_encoder.joblib"
    encoder = joblib.load(encoder_path)
    return encoder


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the test dataset from disk.

    Returns:
        tuple containing:
            pd.DataFrame: Test features
            pd.Series: Test labels
    """
    data_path = "data/processed/test_processed.csv"
    logger.info(f"Loading test data from {data_path}")
    data = pd.read_csv(data_path)
    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y


def evaluate_model(
    model: tf.keras.Model, encoder: LabelEncoder, X: pd.DataFrame, y_true: pd.Series
) -> None:
    """Evaluate the model and generate performance metrics.

    Args:
        model (tf.keras.Model): Trained Keras model.
        encoder (LabelEncoder): Fitted label encoder.
        X (pd.DataFrame): Test features.
        y_true (pd.Series): True labels.
    """
    # Generate model predictions
    y_pred_proba = model.predict(X)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate evaluation metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    evaluation = {"classification_report": report, "confusion_matrix": cm}

    # Log metrics
    logger.info(f"Classification Report:\n{classification_report(y_true, y_pred)}")
    evaluation_path = "metrics/evaluation.json"
    with open(evaluation_path, "w") as f:
        json.dump(evaluation, f, indent=2)


def main() -> None:
    """Main function to orchestrate the model evaluation process."""
    model = load_model()
    encoder = load_encoder()
    X, y = load_test_data()
    evaluate_model(model, encoder, X, y)
    logger.info("Model evaluation completed")


if __name__ == "__main__":
    main()