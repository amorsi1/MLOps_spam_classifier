from src.serve_model import create_prediction_app

# Create app instance for production
app = create_prediction_app(initialize_models_on_startup=True)

if __name__ == "__main__":
    app.run(debug=False)