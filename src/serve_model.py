import mlflow 
# from mlflow import MLflowClient
import pandas as pd
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from flask import Flask, request, jsonify

load_dotenv()
# client = MLflowClient(tracking_uri=os.getenv('MLFLOW_TRACKING_URI')) #configure backend DB

# client = MLflowClient(tracking_uri=MLFLOW_TRACKING_URI)
# client = MLflowClient(tracking_uri=os.getenv('MLFLOW_TRACKING_URI')) #configure backend DB
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI')) #configure backend DB 
mlflow.set_experiment('embedding-spam-ham-classifier')

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')



def load_model_from_registry(model_name: str, version):
    model_name = "lr-best-model"
    model_version = "latest"
    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_version}"

    # # get best model from MLflow
    # best_model = mlflow.search_registered_models(max_results=3,
    #     order_by=['f1_score DESC']
    # )[0]
    # print(best_model)


    model = mlflow.sklearn.load_model(model_uri)
    print(f"Model loaded from registry: {model_name}, version: {version}")
    print(f"Model URI: {model_uri}")
    print(f"Model: {model}")
    return model

#initialize flask app that reponds to pings to EB

app = Flask('vector_spam_classifier')


#on schedule, re-run model training and evaluate if new model is better than the current one. If so register that model and re-initialize the flask model

def encode_text(text, model: SentenceTransformer):
    vector = model.encode(text)
    stacked = vector.reshape(1, -1) #converts vector to 2D array
    return stacked


def main(text ="I am nigerian prince here to give you free money, cocaine, and viagra"):
    model_name = "lr-best-model"
    model_version = "latest"

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    encoded_text = encode_text(text, embedding_model)
    model = load_model_from_registry(model_name, model_version)
    return model.predict(encoded_text)

if __name__ == "__main__":
    main()