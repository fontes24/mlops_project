import logging
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger("src.feature_engineering.engineer_features")


def load_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train and test datasets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets
    """
    train_path = "data/preprocessed/train_preprocessed.csv"
    test_path = "data/preprocessed/test_preprocessed.csv"
    logger.info(f"Loading preprocessed data from {train_path} and {test_path}")
    train_preprocessed = pd.read_csv(train_path)
    test_preprocessed = pd.read_csv(test_path)
    return train_preprocessed, test_preprocessed


def engineer_features(
    train_preprocessed: pd.DataFrame, test_preprocessed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Apply feature engineering transformations.

    Args:
        train_preprocessed (pd.DataFrame): Training dataset
        test_preprocessed (pd.DataFrame): Test dataset

    Returns:
        tuple containing:
            pd.DataFrame: Engineered training features
            pd.DataFrame: Engineered test features
            StandardScaler: Fitted scaler
    """
    logger.info("Engineering features...")
    feature_columns = [col for col in train_preprocessed.columns if col != "target"]

    scaler = StandardScaler()
    
    train_processed = train_preprocessed.copy()
    test_processed = test_preprocessed.copy()

    train_processed[feature_columns] = scaler.fit_transform(train_processed[feature_columns])
    test_processed[feature_columns] = scaler.transform(test_processed[feature_columns])

    return train_processed, test_processed, scaler


def save_artifacts(
    train_processed: pd.DataFrame, test_processed: pd.DataFrame, scaler: StandardScaler
) -> None:
    """Save engineered features and scaler.

    Args:
        train_processed (pd.DataFrame): Engineered training data
        test_processed (pd.DataFrame): Engineered test data
        scaler (StandardScaler): Fitted scaler
    """
    # Save processed data
    output_dir = "data/processed"
    logger.info(f"Saving engineered features to {output_dir}")

    train_path = os.path.join(output_dir, "train_processed.csv")
    test_path = os.path.join(output_dir, "test_processed.csv")

    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)

    # Save scaler
    scaler_path = os.path.join("artifacts", "[features]_scaler.joblib")
    logger.info(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)


def main() -> None:
    """Main function to orchestrate feature engineering pipeline."""
    train_preprocessed, test_preprocessed = load_preprocessed_data()
    train_processed, test_processed, scaler = engineer_features(train_preprocessed, test_preprocessed)
    save_artifacts(train_processed, test_processed, scaler)
    logger.info("Feature engineering completed")


if __name__ == "__main__":
    main()
