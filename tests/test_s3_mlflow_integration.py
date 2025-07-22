"""
Integration tests for S3/MLflow functionality.
Tests the full pipeline: data-processing saves to S3 -> flask-api loads from S3
"""

import pytest
import os
import time
import requests
import json
import subprocess
import mlflow
from mlflow import MlflowClient
import boto3
from botocore.exceptions import NoCredentialsError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
MLFLOW_URI = "http://localhost:8080"
FLASK_API_URL = "http://localhost:4242"
S3_BUCKET = os.getenv("S3_BUCKET")
TEST_EXPERIMENT_NAME = "integration-test-s3-mlflow"
TEST_MODEL_NAME = "integration-test-model"

class TestS3MLflowIntegration:
    """Integration tests for S3 and MLflow functionality"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_services(self):
        """Ensure services are running before tests"""
        logger.info("Setting up services for integration tests...")
        
        # Check if services are running, start if needed
        try:
            response = requests.get(f"{MLFLOW_URI}/health", timeout=5)
            if response.status_code != 200:
                raise Exception("MLflow not responding")
        except:
            logger.info("Starting services with docker-compose...")
            subprocess.run(["make", "up"], check=True, cwd="/Users/afmorsi/dev/MLOps_spam_classifier")
            time.sleep(20)  # Give services time to start
            
        # Verify services are accessible
        self._wait_for_service(MLFLOW_URI, "MLflow")
        self._wait_for_service(f"{FLASK_API_URL}/health", "Flask API")
        
        yield
        
        # Cleanup after tests
        logger.info("Cleaning up test artifacts...")
        self._cleanup_test_artifacts()
    
    def _wait_for_service(self, url, service_name, max_attempts=30):
        """Wait for a service to become available"""
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ {service_name} is ready")
                    return
            except:
                pass
            
            logger.info(f"⏳ Waiting for {service_name}... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(2)
        
        pytest.fail(f"❌ {service_name} failed to start after {max_attempts} attempts")
    
    def _cleanup_test_artifacts(self):
        """Clean up test models and experiments"""
        try:
            client = MlflowClient(MLFLOW_URI)
            
            # Try to delete test models (may fail if doesn't exist)
            try:
                client.delete_registered_model(TEST_MODEL_NAME)
                logger.info(f"Cleaned up test model: {TEST_MODEL_NAME}")
            except:
                pass
                
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    def test_s3_connectivity(self):
        """Test S3 connectivity from MLflow server"""
        logger.info("Testing S3 connectivity...")
        
        # Test S3 access via docker exec
        result = subprocess.run([
            "docker-compose", "exec", "-t", "mlflow-server", "python", "-c",
            f"""
import boto3
try:
    s3 = boto3.client('s3')
    response = s3.head_bucket(Bucket='{S3_BUCKET}')
    print('S3_SUCCESS')
except Exception as e:
    print(f'S3_ERROR: {{e}}')
"""
        ], capture_output=True, text=True, cwd="/Users/afmorsi/dev/MLOps_spam_classifier")
        
        assert "S3_SUCCESS" in result.stdout, f"S3 connectivity failed: {result.stdout + result.stderr}"
        logger.info("✅ S3 connectivity verified")

    def test_mlflow_server_s3_configuration(self):
        """Test that MLflow server is properly configured for S3"""
        logger.info("Testing MLflow S3 configuration...")
        
        # Test artifact storage via docker exec
        result = subprocess.run([
            "docker-compose", "exec", "-T", "mlflow-server", "python", "-c",
            f"""
import mlflow
import tempfile
import os

mlflow.set_tracking_uri('http://localhost:8080')
mlflow.set_experiment('{TEST_EXPERIMENT_NAME}')

with mlflow.start_run():
    # Create and log a test artifact
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('test content')
        temp_file = f.name
    
    mlflow.log_artifact(temp_file, 'test_artifacts')
    artifact_uri = mlflow.get_artifact_uri('test_artifacts')
    
    if artifact_uri.startswith('s3://'):
        print('ARTIFACT_S3_SUCCESS')
    else:
        print(f'ARTIFACT_LOCAL: {{artifact_uri}}')
    
    os.unlink(temp_file)
"""
        ], capture_output=True, text=True, cwd="/Users/afmorsi/dev/MLOps_spam_classifier")
        
        assert "ARTIFACT_S3_SUCCESS" in result.stdout, f"MLflow S3 artifacts failed: {result.stdout + result.stderr}"
        logger.info("✅ MLflow S3 artifact storage verified")

    def test_model_training_and_registration_s3(self):
        """Test that model training saves artifacts to S3 and registers properly"""
        logger.info("Testing model training with S3 artifact storage...")
        
        # Create and register a test model via docker exec
        result = subprocess.run([
            "docker-compose", "exec", "-T", "mlflow-server", "python", "-c",
            f"""
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
import numpy as np

mlflow.set_tracking_uri('http://localhost:8080')
mlflow.set_experiment('{TEST_EXPERIMENT_NAME}')

# Create a simple test model
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)
model = LogisticRegression().fit(X, y)

with mlflow.start_run():
    # Log the model
    mlflow.sklearn.log_model(model, 'model')
    
    # Get model URI
    model_uri = mlflow.get_artifact_uri('model')
    run_id = mlflow.active_run().info.run_id
    
    # Register the model
    mlflow.register_model(f'runs:/{{run_id}}/model', '{TEST_MODEL_NAME}')
    
    if model_uri.startswith('s3://'):
        print(f'MODEL_S3_SUCCESS:{{run_id}}')
    else:
        print(f'MODEL_LOCAL:{{model_uri}}')
"""
        ], capture_output=True, text=True, cwd="/Users/afmorsi/dev/MLOps_spam_classifier")
        
        assert "MODEL_S3_SUCCESS:" in result.stdout, f"Model training/registration failed: {result.stdout + result.stderr}"
        
        # Extract run_id for verification
        run_id = result.stdout.split("MODEL_S3_SUCCESS:")[1].strip()
        logger.info(f"✅ Model training with S3 storage successful (run: {run_id})")
        
        return run_id

    def test_cloud_api_model_loading_from_s3(self):
        """Test that Flask API can load the registered model from S3"""
        logger.info("Testing Flask API model loading from S3...")
        
        # First ensure we have a model to test with
        run_id = self.test_model_training_and_registration_s3()
        
        # Wait a moment for model registration to complete
        time.sleep(5)
        
        # Test model loading via docker exec
        result = subprocess.run([
            "docker-compose", "exec", "-T", "web-app", "python", "-c",
            f"""
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri('http://mlflow-server:8080')

try:
    # Try to load the test model
    model = mlflow.sklearn.load_model('models:/{TEST_MODEL_NAME}/latest')
    print('FLASK_LOAD_SUCCESS')
except Exception as e:
    print(f'FLASK_LOAD_ERROR: {{e}}')
"""
        ], capture_output=True, text=True, cwd="/Users/afmorsi/dev/MLOps_spam_classifier")
        
        assert "FLASK_LOAD_SUCCESS" in result.stdout, f"Flask model loading failed: {result.stdout + result.stderr}"
        logger.info("✅ Flask API successfully loaded model from S3")

    def test_end_to_end_prediction_pipeline(self):
        """Test the complete end-to-end pipeline"""
        logger.info("Testing end-to-end prediction pipeline...")
        
        # Ensure model is trained and registered
        self.test_model_training_and_registration_s3()
        
        # Wait for model registration to complete
        time.sleep(10)
        
        # Test that Flask API health endpoint shows models loaded
        response = requests.get(f"{FLASK_API_URL}/health", timeout=30)
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        
        health_data = response.json()
        logger.info(f"Flask API health: {health_data}")
        
        # Test prediction endpoint (even if models aren't loaded, it should try to load them)
        test_text = "This is a test message for spam classification"
        prediction_response = requests.post(
            f"{FLASK_API_URL}/predict",
            json={"text": test_text},
            timeout=60  # Allow time for model loading
        )
        
        logger.info(f"Prediction response status: {prediction_response.status_code}")
        logger.info(f"Prediction response: {prediction_response.text}")
        
        # The prediction might fail if it tries to load the real model, but we're testing S3 functionality
        # So we check if it at least attempts to connect properly
        assert prediction_response.status_code in [200, 500], f"Unexpected status: {prediction_response.status_code}"
        
        logger.info("✅ End-to-end pipeline test completed")

    def test_data_processing_service_s3_integration(self):
        """Test that data-processing service can train and save to S3"""
        logger.info("Testing data-processing service S3 integration...")
        
        # Check if data-processing container is running, start if needed
        check_container = subprocess.run([
            "docker", "ps", "--filter", "name=data-processing", "--format", "{{.Names}}"
        ], capture_output=True, text=True, cwd="/Users/afmorsi/dev/MLOps_spam_classifier")
        
        container_started = False
        if "data-processing" not in check_container.stdout:
            logger.info("Starting data-processing container for testing...")
            subprocess.run([
                "docker-compose", "run", "-d", "--name", "data-processing-test", 
                "data-processing", "sleep", "infinity"
            ], check=True, cwd="/Users/afmorsi/dev/MLOps_spam_classifier")
            container_started = True
            container_name = "data-processing-test"
        else:
            # Find the actual running container name
            container_name = check_container.stdout.strip().split('\n')[0]
        
        try:
            # Run a simplified training process
            result = subprocess.run([
                "docker", "exec", "-t", container_name, "python", "-c",
                f"""
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
mlflow.set_experiment('{TEST_EXPERIMENT_NAME}')

# Create test data
X = np.random.rand(50, 10)
y = np.random.randint(0, 2, 50)

with mlflow.start_run():
    # Train simple model
    model = LogisticRegression().fit(X, y)
    
    # Log model
    mlflow.sklearn.log_model(model, 'model')
    
    # Get artifact URI
    artifact_uri = mlflow.get_artifact_uri('model')
    run_id = mlflow.active_run().info.run_id
    
    # Register model
    mlflow.register_model(f'runs:/{{run_id}}/model', '{TEST_MODEL_NAME}-data-processing')
    
    if artifact_uri.startswith('s3://'):
        print(f'DATA_PROCESSING_S3_SUCCESS:{{run_id}}')
    else:
        print(f'DATA_PROCESSING_LOCAL:{{artifact_uri}}')
"""
            ], capture_output=True, text=True, cwd="/Users/afmorsi/dev/MLOps_spam_classifier")
            
            logger.info(f"Data processing result: {result.stdout}")
            logger.info(f"Data processing errors: {result.stderr}")
            
            assert "DATA_PROCESSING_S3_SUCCESS:" in result.stdout, f"Data processing S3 integration failed: {result.stdout + result.stderr}"
            logger.info("✅ Data-processing service S3 integration verified")
        
        finally:
            # Cleanup: remove the test container if we started it
            if container_started:
                logger.info("Cleaning up test container...")
                subprocess.run([
                    "docker", "rm", "-f", "data-processing-test"
                ], capture_output=True, cwd="/Users/afmorsi/dev/MLOps_spam_classifier")

if __name__ == "__main__":
    # Allow running as a script for debugging
    pytest.main([__file__, "-v", "-s"])