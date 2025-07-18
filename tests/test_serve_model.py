import sys
import os
import pytest
from unittest.mock import Mock, patch

# # Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from serve_model import create_prediction_app, ModelManager

@pytest.fixture
def model_manager():
    return ModelManager()

def test_initialization(model_manager):
    """Test ModelManager initialization"""
    assert model_manager.embedding_model is None, "newly initialized model should not have this ready"
    assert model_manager.prediction_model is None, "newly initialized model should not have this ready"
    assert not model_manager._initialized, "newly initialized model should not be intialized"
    assert not model_manager.is_ready(), "newly initialized model should not be ready"

@patch('serve_model.SentenceTransformer')
@patch('serve_model.mlflow.sklearn.load_model')
def test_initialize_models(mock_load_model, mock_sentence_transformer, model_manager):
    """Test model initialization"""
    mock_embedding_model = Mock()
    mock_prediction_model = Mock()
    mock_sentence_transformer.return_value = mock_embedding_model
    mock_load_model.return_value = mock_prediction_model

    model_manager.initialize_models()

    assert model_manager._initialized
    assert model_manager.embedding_model == mock_embedding_model
    assert model_manager.prediction_model == mock_prediction_model
    assert model_manager.is_ready()

def test_encode_text_without_model(model_manager):
    """Test encode_text raises error when model not loaded"""
    with pytest.raises(ValueError):
        model_manager.encode_text("test text")

@pytest.fixture
def flask_app():
    app = create_prediction_app(initialize_models_on_startup=False)
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(flask_app):
    return flask_app.test_client()

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200

    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['service'] == 'spam-classifier'
    assert 'models_loaded' in data

def test_predict_endpoint_missing_data(client):
    """Test predict endpoint with missing data"""
    response = client.post('/predict', json={})
    assert response.status_code == 400

    data = response.get_json()
    assert 'error' in data
    assert 'Missing' in data['error']

def test_predict_endpoint_invalid_text(client):
    """Test predict endpoint with invalid text"""
    response = client.post('/predict', json={'text': ''})
    assert response.status_code == 400

    data = response.get_json()
    assert 'error' in data
    assert 'non-empty string' in data['error']

def test_predict_endpoint_non_string_text(client):
    """Test predict endpoint with non-string text"""
    response = client.post('/predict', json={'text': 123})
    assert response.status_code == 400

    data = response.get_json()
    assert 'error' in data
    assert 'non-empty string' in data['error']

@patch('serve_model.ModelManager.predict')
def test_predict_endpoint_success(mock_predict, client, flask_app):
    """Test predict endpoint with valid input"""
    mock_predict.return_value = {
        "prediction": 1,
        "prediction_label": "spam",
        "confidence": {"ham": 0.1, "spam": 0.9},
        "embedding": [0.1, 0.2, 0.3]
    }
    # Patch the model_manager in the app context
    with flask_app.app_context():
        flask_app.model_manager = ModelManager()
        flask_app.model_manager.predict = mock_predict

        response = client.post('/predict', json={'text': 'hello world'})
        assert response.status_code == 200
        data = response.get_json()
        assert data['text'] == 'hello world'
        assert data['prediction'] == 1
        assert data['prediction_label'] == 'spam'
        assert 'confidence' in data
        assert 'embedding' in data

def test_health_endpoint_models_loaded(client, flask_app):
    """Test health endpoint when models are loaded"""
    with flask_app.app_context():
        flask_app.model_manager._initialized = True
        flask_app.model_manager.embedding_model = Mock()
        flask_app.model_manager.prediction_model = Mock()
        response = client.get('/health')
        data = response.get_json()
        assert data['models_loaded'] is True
        
        # Additional checks for the health endpoint
        assert data['status'] == 'healthy'
        assert data['service'] == 'spam-classifier'
        assert 'models_loaded' in data

def test_predict_endpoint_missing_data_alternative(client):
    """Test predict endpoint with missing data - alternative test"""
    response = client.post('/predict', json={})
    assert response.status_code == 400
    
    data = response.get_json()
    assert 'error' in data
    assert 'Missing' in data['error']

def test_predict_endpoint_invalid_text_alternative(client):
    """Test predict endpoint with invalid text - alternative test"""
    response = client.post('/predict', json={'text': ''})
    assert response.status_code == 400
    
    data = response.get_json()
    assert 'error' in data
    assert 'non-empty string' in data['error']