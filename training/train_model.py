"""
Car Price Prediction Model Training Script

This script loads the car pricing data, preprocesses it, trains a regression model,
and saves the trained model for use in the FastAPI service.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Get the project root directory (parent of training folder)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
DATA_FILE = os.getenv("DATA_FILE", "car_pricing_amjad_zhour.csv")
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = "car_price_model.joblib"
RANDOM_STATE = 42


def load_data():
    """Load the car pricing dataset."""
    data_path = DATA_DIR / DATA_FILE
    print(f"Loading data from: {data_path}")

    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {data_path}")


def explore_data(df):
    """Print basic information about the dataset."""
    print("\n--- DATA OVERVIEW ---")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- TARGET VARIABLE (Price) STATISTICS ---")
    print(df["Price"].describe())

    # Check for categorical variables
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print("\n--- CATEGORICAL COLUMNS ---")
    print(categorical_cols)

    for col in categorical_cols:
        print(f"\nUnique values in '{col}': {df[col].nunique()}")
        print(df[col].value_counts().head())


def preprocess_data(df):
    """
    Preprocess the data for model training.

    Returns:
        X: Feature matrix
        y: Target vector
        preprocessor: Fitted preprocessing pipeline
    """
    print("\n--- PREPROCESSING DATA ---")

    # Define features and target
    target = "Price"

    # Drop target from features
    X = df.drop(columns=[target])
    y = df[target]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    return X, y, preprocessor


def create_model():
    """Create and return the machine learning model."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return model


def train_model(X, y, preprocessor, model):
    """
    Train the model with preprocessing pipeline.

    Returns:
        pipeline: Trained pipeline (preprocessor + model)
        X_test: Test features
        y_test: Test targets
    """
    print("\n--- TRAINING MODEL ---")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Create complete pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Train the model
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test


def evaluate_model(pipeline, X_test, y_test):
    """Evaluate the trained model."""
    print("\n--- MODEL EVALUATION ---")

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Show some example predictions
    print("\n--- EXAMPLE PREDICTIONS ---")
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        print(
            f"Actual: ${actual:,.2f}, Predicted: ${predicted:,.2f}, Diff: ${abs(actual - predicted):,.2f}"
        )

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def save_model(pipeline, metrics, feature_names):
    """Save the trained model and metadata."""
    print("\n--- SAVING MODEL ---")

    # Create models directory if it doesn't exist
    MODEL_DIR.mkdir(exist_ok=True)

    # Model path
    model_path = MODEL_DIR / MODEL_FILE

    # Save the pipeline
    joblib.dump(pipeline, model_path)

    # Save model metadata
    metadata = {
        "model_type": "RandomForestRegressor",
        "feature_names": feature_names,
        "metrics": metrics,
        "model_file": MODEL_FILE,
        "trained_on": pd.Timestamp.now().isoformat(),
    }

    metadata_path = MODEL_DIR / "model_metadata.joblib"
    joblib.dump(metadata, metadata_path)

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

    return model_path, metadata_path


def main():
    """Main training function."""
    print("=== CAR PRICE PREDICTION MODEL TRAINING ===")

    try:
        # Load data
        df = load_data()

        # Explore data
        explore_data(df)

        # Preprocess data
        X, y, preprocessor = preprocess_data(df)

        # Create model
        model = create_model()

        # Train model
        pipeline, X_test, y_test = train_model(X, y, preprocessor, model)

        # Evaluate model
        metrics = evaluate_model(pipeline, X_test, y_test)

        # Save model
        feature_names = X.columns.tolist()
        model_path, metadata_path = save_model(pipeline, metrics, feature_names)

        print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
        print("Model Performance Summary:")
        print(f"  - R² Score: {metrics['r2']:.4f}")
        print(f"  - RMSE: ${metrics['rmse']:,.2f}")
        print(f"  - MAE: ${metrics['mae']:,.2f}")
        print(f"Model saved at: {model_path}")

    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
