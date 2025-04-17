from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uvicorn
import logging
from app.model import MLModel
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="API for making predictions with a pre-trained ML model",
    version="1.0.0"
)

# Create model instance
model = MLModel()

class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""
    features: List[float] = Field(..., example=[5.1, 3.5, 1.4, 0.2])

class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""
    prediction: int = Field(..., example=0)
    probabilities: List[float] = Field(..., example=[0.9, 0.05, 0.05])

class HealthResponse(BaseModel):
    """Response from health check endpoint."""
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)

@asynccontextmanager
async def lifespan():
    """Load the ML model when the API starts."""
    logger.info("Loading ML model on startup")
    model.load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = model.model is not None
    if not model_loaded:
        # Try to load the model if it's not already loaded
        model_loaded = model.load_model()
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

@app.post("/predict", response_model=Dict[str, Any])
async def predict(request: PredictionRequest):
    """Make a prediction using the ML model."""
    try:
        result = model.predict(request.features)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# For local development
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)