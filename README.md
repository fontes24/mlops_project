# ML Classifier - Breast Cancer Prediction

A machine learning project that builds and deploys a neural network classifier for breast cancer prediction using the scikit-learn breast cancer dataset.

## Project Structure

```
/mlops_project
├── app/                          # Web application
│   ├── __init__.py               # Package marker
│   ├── main.py                   # Flask app with prediction API
│   └── templates/                # HTML templates for web UI
│       └── index.html            # Web interface for predictions
├── artifacts/                    # Preprocessing artifacts
├── data/                         # Data storage
│   ├── preprocessed/             # Cleaned data
│   ├── processed/                # Feature-engineered data
│   └── raw/                      # Raw dataset
├── metrics/                      # Model performance metrics
├── models/                       # Trained model
├── src/                          # Source code modules
│   ├── __init__.py               # Package marker
│   ├── data_loading/             # Data loading utilities
│   │   ├── __init__.py           # Package marker
│   │   └── load_data.py          # Dataset loading and preparation
│   ├── data_preprocessing/       # Data cleaning and splitting
│   │   ├── __init__.py           # Package marker
│   │   └── preprocess_data.py    # Data cleaning and imputation
│   ├── feature_engineering/      # Feature transformation utilities
│   │   ├── __init__.py           # Package marker
│   │   └── engineer_features.py  # Feature scaling and transformation
│   ├── model_evaluation/         # Model evaluation scripts
│   │   ├── __init__.py           # Package marker
│   │   └── evaluate_model.py     # Model performance evaluation
│   └── model_training/           # Model training scripts
│       ├── __init__.py           # Package marker
│       └── train_model.py        # Neural network training
├── .dockerignore                 # Docker ignore rules
├── Dockerfile                    # Docker build instructions
├── params.yaml                   # Configuration parameters
├── pyproject.toml                # Python dependencies and project metadata
└── README.md                     # Project documentation
```

## Features

- **Data Pipeline**: Complete ETL pipeline from raw data to model-ready features
- **Neural Network**: TensorFlow/Keras deep learning model with configurable architecture
- **Web Interface**: Flask-based web application for making predictions
- **Artifact Management**: Serialized models and preprocessors for deployment
- **Evaluation Metrics**: Comprehensive model performance analysis

## Dependencies

The project requires Python 3.12+ and the packages informed in `pyproject.toml`.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mlops_project
```

2. Install dependencies:
```bash
pip install -e .
```

## Configuration

Model hyperparameters and data processing settings are configured in `params.yaml`.

## Model Architecture

The neural network consists of a multilayer perceptron with 2 hidden layers.

## Artifacts

The training process generates the following files:

In the `models/` directory:
- `model.keras`: Trained TensorFlow model

In the `artifacts/` directory:
- `[features]_mean_imputer.joblib`: Feature imputer for missing values
- `[features]_scaler.joblib`: Standard scaler for feature normalization
- `[target]_one_hot_encoder.joblib`: One-hot encoder for target labels

## Metrics

Model performance metrics are saved to:
- `metrics/training.json`: Training history and validation metrics
- `metrics/evaluation.json`: Test set performance and confusion matrix

## Development

The project follows a modular structure with separate concerns:
- **Data Loading**: Fetches and saves raw breast cancer dataset
- **Preprocessing**: Handles missing values and data splitting
- **Feature Engineering**: Applies scaling transformations
- **Model Training**: Builds and trains the neural network
- **Model Evaluation**: Generates performance metrics
- **Web Application**: Provides prediction interface

Each module can be run independently and saves its outputs for the next stage in the pipeline.

## Usage

### Training the Model

Run the complete ML pipeline (for proper logging to the terminal, run as modules with `python -m`):

```bash
# 1. Load and prepare raw data
python -m src.data_loading.load_data

# 2. Preprocess data (imputation, train/test split)
python -m src.data_preprocessing.preprocess_data

# 3. Engineer features (scaling)
python -m src.feature_engineering.engineer_features

# 4. Train the neural network model
python -m src.model_training.train_model

# 5. Evaluate model performance
python -m src.model_evaluation.evaluate_model
```

### Running the Web Application

#### Flask

After training the model, start the Flask web server:

```bash
python app/main.py
```

The application will be available at `http://localhost:5001`

### Docker

You can instead build and run the application using Docker:

#### Build the Docker image

```bash
docker build -t ml-classifier .
```

#### Run the Docker container

```bash
docker run -p 5001:5001 ml-classifier
```

The web application will be available at `http://localhost:5001`.

### Making Predictions

1. **Web Interface**: Upload a CSV file with breast cancer features through the web interface
2. **API**: The `/upload` endpoint accepts CSV files and returns predictions

#### Required CSV Format

Your CSV file must contain all 30 breast cancer features with exact column names:
- mean radius, mean texture, mean perimeter, mean area, mean smoothness, etc.
- See `sklearn.datasets.load_breast_cancer().feature_names` for the complete list
