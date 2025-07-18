# MLOps Spam Classifier Makefile
# Usage: make <target>

.PHONY: help up down build test-api test-health test-spam test-ham evaluate-response clean logs test-s3-integration test-pipeline

# Default target
help:
	@echo "Available targets:"
	@echo "  up                    - Start all services with docker-compose"
	@echo "  down                  - Stop all services"
	@echo "  build                 - Build all Docker images"
	@echo "  test-health           - Test health endpoint"
	@echo "  test-spam             - Test prediction with spam text"
	@echo "  test-ham              - Test prediction with ham text"
	@echo "  test-api              - Run all API tests"
	@echo "  test-s3-integration   - Run S3/MLflow integration tests"
	@echo "  test-pipeline         - Run complete pipeline tests (includes S3)"
	@echo "  evaluate-response     - Evaluate if API is responding correctly"
	@echo "  logs                  - Show logs from all services"
	@echo "  clean                 - Clean up Docker resources"

# Docker Compose targets
up:
	@echo "Starting services..."
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Services should be ready at:"
	@echo "  MLflow: http://localhost:8080"
	@echo "  Flask API: http://localhost:4242"

down:
	@echo "Stopping services..."
	docker-compose down

build:
	@echo "Building Docker images..."
	docker-compose build

# API Testing targets
test-health:
	@echo "Testing health endpoint..."
	@curl -s http://localhost:4242/health | python -m json.tool

test-spam:
	@echo "Testing spam prediction..."
	@curl -s -X POST http://localhost:4242/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "FREE VIAGRA! Click here now to get your free pills! Limited time offer!"}'

test-ham:
	@echo "Testing ham prediction..."
	@curl -s -X POST http://localhost:4242/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "Hi, how are you doing today? Hope you have a great day!"}'

test-api: test-health test-spam test-ham
	@echo "All API tests completed"

# Response evaluation target
evaluate-response:
	@echo "Evaluating API response..."
	@RESPONSE=$$(curl -s -X POST http://localhost:4242/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "This is a test message"}' 2>/dev/null) && \
	if echo "$$RESPONSE" | python -c "import sys, json; data=json.load(sys.stdin); print('✅ Valid JSON response'); print('✅ Text field:', data.get('text', 'MISSING')); print('✅ Prediction:', data.get('prediction_label', 'MISSING')); print('✅ Confidence:', data.get('confidence', {}).get('spam', 'MISSING'))" 2>/dev/null; then \
		echo "🎉 API is responding correctly!"; \
	else \
		echo "❌ API response evaluation failed"; \
		echo "Response: $$RESPONSE"; \
	fi

# Utility targets
logs:
	@echo "Showing logs from all services..."
	docker-compose logs -f

clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f

# Development targets
dev-up: up
	@echo "Development environment ready!"
	@echo "Run 'make test-api' to test the API"

# Full integration test
integration-test: up
	@echo "Running integration tests..."
	@sleep 15
	@make evaluate-response
	@make test-api

# S3/MLflow integration test
test-s3-integration:
	@echo "Running S3/MLflow integration tests..."
	@pytest tests/test_s3_mlflow_integration.py -v -s

# Complete pipeline test including S3
test-pipeline: up
	@echo "Running complete pipeline tests..."
	@sleep 20
	@make test-s3-integration