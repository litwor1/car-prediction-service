"""
Pydantic models for request/response validation.
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class FuelType(str, Enum):
    """Allowed fuel types."""

    PETROL = "Petrol"
    DIESEL = "Diesel"
    ELECTRIC = "Electric"


class Transmission(str, Enum):
    """Allowed transmission types."""

    MANUAL = "Manual"
    AUTOMATIC = "Automatic"


class Make(str, Enum):
    """Allowed car makes."""

    HONDA = "Honda"
    FORD = "Ford"
    BMW = "BMW"
    AUDI = "Audi"
    TOYOTA = "Toyota"


class Model(str, Enum):
    """Allowed car models."""

    MODEL_A = "Model A"
    MODEL_B = "Model B"
    MODEL_C = "Model C"
    MODEL_D = "Model D"
    MODEL_E = "Model E"


class CarPredictionRequest(BaseModel):
    """Request model for car price prediction."""

    make: Make = Field(..., description="Car manufacturer")
    model: Model = Field(..., description="Car model")
    year: int = Field(..., ge=1990, le=2030, description="Manufacturing year")
    engine_size: float = Field(..., ge=1.0, le=6.0, description="Engine size in liters")
    mileage: int = Field(..., ge=0, le=500000, description="Car mileage in kilometers")
    fuel_type: FuelType = Field(..., description="Type of fuel")
    transmission: Transmission = Field(..., description="Transmission type")

    @validator("year")
    def validate_year(cls, v):
        if v < 1990 or v > 2030:
            raise ValueError("Year must be between 1990 and 2030")
        return v

    @validator("engine_size")
    def validate_engine_size(cls, v):
        if v <= 0:
            raise ValueError("Engine size must be positive")
        return v

    @validator("mileage")
    def validate_mileage(cls, v):
        if v < 0:
            raise ValueError("Mileage cannot be negative")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "make": "Honda",
                "model": "Model B",
                "year": 2020,
                "engine_size": 2.5,
                "mileage": 50000,
                "fuel_type": "Petrol",
                "transmission": "Automatic",
            }
        }

    def to_model_format(self) -> Dict[str, Any]:
        """Convert to format expected by the ML model."""
        return {
            "Make": self.make.value,
            "Model": self.model.value,
            "Year": self.year,
            "Engine Size": self.engine_size,
            "Mileage": self.mileage,
            "Fuel Type": self.fuel_type.value,
            "Transmission": self.transmission.value,
        }


class CarPredictionResponse(BaseModel):
    """Response model for car price prediction."""

    predicted_price: float = Field(
        ..., description="Predicted car price in currency units"
    )
    model_version: str = Field(..., description="Version of the model used")
    model_r2_score: Optional[float] = Field(None, description="Model R² score")
    input_features: Dict[str, Any] = Field(
        ..., description="Input features used for prediction"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field("healthy", description="Service status")
    message: str = Field(..., description="Status message")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_type: Optional[str] = Field(None, description="Type of ML model")
    feature_names: Optional[list] = Field(None, description="Names of features used")
    metrics: Optional[Dict[str, float]] = Field(
        None, description="Model performance metrics"
    )
    trained_on: Optional[str] = Field(None, description="When the model was trained")
    status: str = Field(..., description="Model loading status")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
