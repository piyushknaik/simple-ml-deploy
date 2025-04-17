import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")

class MLModel:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """Load the pre-trained model from disk."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # If model doesn't exist, train a simple one (for demo purposes)
            logger.info("Training a simple model instead")
            self._train_simple_model()
            return True
            
    def _train_simple_model(self):
        """Train a simple model for demonstration purposes."""
        logger.info("Training a simple RandomForest model on iris dataset")
        # Load iris dataset
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        
        # Train a simple RandomForest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Simple model trained and saved to {self.model_path}")
    
    def predict(self, features):
        """Make predictions using the loaded model."""
        if self.model is None:
            success = self.load_model()
            if not success:
                return {"error": "Model could not be loaded"}
        
        try:
            # Convert input to numpy array
            features_array = np.array(features, dtype=float)
            
            # Make prediction
            prediction = self.model.predict(features_array.reshape(1, -1))
            prediction_proba = self.model.predict_proba(features_array.reshape(1, -1))
            
            # Format the response
            result = {
                "prediction": int(prediction[0]),
                "probabilities": prediction_proba[0].tolist()
            }
            return result
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}