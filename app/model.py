"""
Model loading and prediction logic for the car price prediction service.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the project root directory
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = "car_price_model.joblib"
METADATA_FILE = "model_metadata.joblib"


class CarPricePredictor:
    """Car price prediction model wrapper."""

    def __init__(self):
        """Initialize the predictor and load the model."""
        self.model = None
        self.metadata = None
        self.feature_names = None
        self._load_model()

    def _load_model(self):
        """Load the trained model and metadata."""
        try:
            model_path = MODEL_DIR / MODEL_FILE
            metadata_path = MODEL_DIR / METADATA_FILE

            logger.info(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)

            logger.info(f"Loading metadata from: {metadata_path}")
            self.metadata = joblib.load(metadata_path)
            self.feature_names = self.metadata.get("feature_names", [])

            logger.info("Model loaded successfully!")
            logger.info(f"Model type: {self.metadata.get('model_type', 'Unknown')}")
            logger.info(
                f"Model R² score: {self.metadata.get('metrics', {}).get('r2', 'Unknown')}"
            )

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, car_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a price prediction for a given car.

        Args:
            car_data: Dictionary containing car features

        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([car_data])

            # Ensure all expected columns are present
            expected_features = [
                "Make",
                "Model",
                "Year",
                "Engine Size",
                "Mileage",
                "Fuel Type",
                "Transmission",
            ]
            for feature in expected_features:
                if feature not in input_df.columns:
                    # Handle missing features with defaults or raise error
                    if feature in ["Year", "Engine Size", "Mileage"]:
                        raise ValueError(
                            f"Required numerical feature '{feature}' is missing"
                        )
                    else:
                        # For categorical features, you might want to set a default
                        logger.warning(
                            f"Feature '{feature}' not provided, using default"
                        )

            # Make prediction
            prediction = self.model.predict(input_df)[0]

            # Ensure prediction is positive (car prices can't be negative)
            prediction = max(0, prediction)

            result = {
                "predicted_price": round(prediction, 2),
                "model_version": self.metadata.get("model_file", "unknown"),
                "model_r2_score": self.metadata.get("metrics", {}).get("r2", None),
                "input_features": car_data,
            }

            logger.info(f"Prediction made: ${prediction:.2f} for {car_data}")
            return result

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.metadata is None:
            return {"error": "Model metadata not available"}

        return {
            "model_type": self.metadata.get("model_type"),
            "feature_names": self.feature_names,
            "metrics": self.metadata.get("metrics", {}),
            "trained_on": self.metadata.get("trained_on"),
            "status": "loaded",
        }


# Global predictor instance
predictor = CarPricePredictor()
