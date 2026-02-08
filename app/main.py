"""
Car Price Prediction FastAPI Service

A simple FastAPI service that provides car price predictions using a trained ML model.
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import predictor
from .schemas import (
    CarPredictionRequest,
    CarPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Car Price Prediction Service",
    description="A machine learning service that predicts car prices based on various features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Startup event to initialize the model."""
    logger.info("Starting Car Price Prediction Service...")
    try:
        # Model is already loaded in the predictor instance
        model_info = predictor.get_model_info()
        logger.info(f"Model loaded: {model_info.get('model_type', 'Unknown')}")
        logger.info("Service started successfully!")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise


@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the service is running and healthy",
)
async def root():
    """Root endpoint - health check."""
    try:
        model_info = predictor.get_model_info()
        if model_info.get("status") == "loaded":
            return HealthResponse(
                status="healthy",
                message="Car Price Prediction Service is running and model is loaded",
            )
        else:
            return HealthResponse(
                status="unhealthy",
                message="Service is running but model is not loaded properly",
            )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service health check failed")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Detailed Health Check",
    description="Detailed health check including model status",
)
async def health_check():
    """Detailed health check endpoint."""
    try:
        model_info = predictor.get_model_info()
        return HealthResponse(
            status="healthy" if model_info.get("status") == "loaded" else "unhealthy",
            message=f"Model status: {model_info.get('status', 'unknown')}",
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Get information about the loaded ML model",
)
async def get_model_info():
    """Get model information."""
    try:
        model_info = predictor.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve model information"
        )


@app.post(
    "/predict",
    response_model=CarPredictionResponse,
    summary="Predict Car Price",
    description="Predict the price of a car based on its features",
)
async def predict_car_price(car_data: CarPredictionRequest):
    """
    Predict car price based on input features.

    - **make**: Car manufacturer (Honda, Ford, BMW, Audi, Toyota)
    - **model**: Car model (Model A, Model B, Model C, Model D, Model E)
    - **year**: Manufacturing year (1990-2030)
    - **engine_size**: Engine size in liters (1.0-6.0)
    - **mileage**: Car mileage in kilometers (0-500,000)
    - **fuel_type**: Type of fuel (Petrol, Diesel, Electric)
    - **transmission**: Transmission type (Manual, Automatic)
    """
    try:
        logger.info(f"Received prediction request: {car_data}")

        # Convert Pydantic model to format expected by ML model
        model_input = car_data.to_model_format()

        # Make prediction
        prediction_result = predictor.predict(model_input)

        # Return response
        response = CarPredictionResponse(**prediction_result)
        logger.info(
            f"Prediction successful: ${prediction_result['predicted_price']:.2f}"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction service error")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return {"error": exc.detail}


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "Internal server error"}


# Additional utility endpoints
@app.get(
    "/features",
    summary="Available Features",
    description="Get information about available car features for prediction",
)
async def get_available_features():
    """Get available options for car features."""
    return {
        "makes": ["Honda", "Ford", "BMW", "Audi", "Toyota"],
        "models": ["Model A", "Model B", "Model C", "Model D", "Model E"],
        "fuel_types": ["Petrol", "Diesel", "Electric"],
        "transmissions": ["Manual", "Automatic"],
        "year_range": {"min": 1990, "max": 2030},
        "engine_size_range": {"min": 1.0, "max": 6.0},
        "mileage_range": {"min": 0, "max": 500000},
    }
