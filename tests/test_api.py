import pytest
from fastapi.testclient import TestClient
import numpy as np
from app.main import app
from app.model import MLModel

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_endpoint():
    """Test the prediction endpoint with valid data."""
    # Sample iris flower features (sepal length, sepal width, petal length, petal width)
    features = [5.1, 3.5, 1.4, 0.2]  # This is a Setosa
    
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probabilities" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probabilities"], list)
    assert len(data["probabilities"]) > 0

def test_predict_endpoint_invalid_data():
    """Test the prediction endpoint with invalid data."""
    # Invalid input - string instead of float
    response = client.post("/predict", json={"features": ["invalid", 3.5, 1.4, 0.2]})
    assert response.status_code == 500

    # Invalid input - missing features
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_model_class():
    """Test the MLModel class directly."""
    model = MLModel()
    assert model.load_model() == True
    
    # Test prediction
    features = np.array([5.1, 3.5, 1.4, 0.2])
    result = model.predict(features)
    assert "prediction" in result
    assert "probabilities" in result