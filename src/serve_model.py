import mlflow 
from dotenv import load_dotenv
import os
import time
from sentence_transformers import SentenceTransformer
import numpy as np
from flask import Flask, request, jsonify, current_app

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI')) 
mlflow.set_experiment('embedding-spam-ham-classifier-s3')

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')

class ModelManager:
    """Manages model loading and caching"""
    
    def __init__(self):
        self.embedding_model = None
        self.prediction_model = None
        self._initialized = False
    
    def load_model_from_registry(self, model_name: str = "lr-best-model", version: str = "latest"):
        """Load ML model from MLflow registry"""
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        model_uri = f"models:/{model_name}/{version}"
        print(f"Attempting to load model URI: {model_uri}")
        
        # Debug: Check what registered models exist
        try:
            client = mlflow.MlflowClient()
            models = client.search_registered_models()
            print(f"Available registered models: {[m.name for m in models]}")
        except Exception as e:
            print(f"Failed to list registered models: {e}")
        
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded from registry: {model_name}, version: {version}")
        return model
    
    def encode_text(self, text: str):
        """Convert text to embedding vector"""
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded")
        
        vector = self.embedding_model.encode(text)
        return vector.reshape(1, -1)  # converts vector to 2D array
    
    def initialize_models(self):
        """Initialize models on startup"""
        if self._initialized:
            return
        
        print("Loading models...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.prediction_model = self.load_model_from_registry()
        self._initialized = True
        print("Models loaded successfully!")
    
    def is_ready(self):
        """Check if models are loaded and ready"""
        return self._initialized and self.embedding_model is not None and self.prediction_model is not None
    
    def predict(self, text: str):
        """Make prediction on text"""
        if not self.is_ready():
            self.initialize_models()
        
        embedding = self.encode_text(text)
        prediction = self.prediction_model.predict(embedding)[0]
        prediction_proba = self.prediction_model.predict_proba(embedding)[0]
        
        return {
            "prediction": int(prediction),
            "prediction_label": "spam" if prediction == 1 else "ham",
            "confidence": {
                "ham": float(prediction_proba[0]),
                "spam": float(prediction_proba[1])
            },
            "embedding": embedding.tolist()[0]
        }

def health_check():
    """Health check endpoint"""
    model_manager = current_app.model_manager
    return jsonify({
        "status": "healthy",
        "service": "spam-classifier",
        "models_loaded": model_manager.is_ready()
    })

def predict():
    """Predict endpoint that takes raw text and returns prediction + embedding"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        
        if not text or not isinstance(text, str):
            return jsonify({"error": "Text must be a non-empty string"}), 400
        
        # Get model manager from app context
        model_manager = current_app.model_manager
        
        # Make prediction
        result = model_manager.predict(text)
        
        # Prepare response
        response = {
            "text": text,
            **result
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

def create_prediction_app(initialize_models_on_startup=True, max_attempts:int = 6):
    """Application factory pattern for creating Flask app"""
    app = Flask('vector_spam_classifier')
    
    # Initialize model manager
    app.model_manager = ModelManager()
    
    # Initialize models if requested
    if initialize_models_on_startup:
        with app.app_context(): 
            for attempt in range(max_attempts):
                try:
                    app.model_manager.initialize_models()
                    break
                except Exception as e:
                    if attempt < max_attempts - 1:
                        print(f"⏳ Waiting for models... (attempt {attempt + 1}/{max_attempts})")
                        time.sleep(10)
                    else:
                        print(f"Failed to initialize models on startup: {e}")
                        print("App will start without models - they can be loaded on first request")
    
    # Register routes
    app.add_url_rule('/health', 'health_check', health_check, methods=['GET'])
    app.add_url_rule('/predict', 'predict', predict, methods=['POST'])
    
    return app

def main(text="I am nigerian prince here to give you free money, cocaine, and viagra"):
    """Test function for development"""
    model_manager = ModelManager()
    model_manager.initialize_models()
    
    result = model_manager.predict(text)
    
    print(f"Text: {text}")
    print(f"Prediction: {result['prediction']} ({result['prediction_label']})")
    print(f"Confidence: {result['confidence']}")
    
    return result

if __name__ == "__main__":
    # For development/testing
    app = create_prediction_app()
    app.run(host='0.0.0.0', port=4242, debug=True)