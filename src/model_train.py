import mlflow
import os
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pickle 
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from dotenv import load_dotenv
import uuid

load_dotenv()  # Load environment variables from .env file

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI')) #configure backend DB 
mlflow.set_experiment('embedding-spam-ham-classifier')
mlflow.sklearn.autolog()  # Automatically log parameters, metrics, and models

def load_data(combined_df_path: str):
# def combine_datasets(List[pd.DataFrame]):
    with open(combined_df_path, 'rb') as f:
        df = pickle.load(f)
    return df

def separate_features_and_target(df, target='label', features=['embedding']):
    y = df[target]
    X = df[features]
    
    # Convert embeddings to proper numpy array format
    if 'embedding' in X.columns:
        # Stack all embeddings into a 2D array
        embeddings = np.vstack(X['embedding'].values)
        return y, embeddings
    
    return y, X

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the DataFrame into training and testing sets.
    """
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
    return separate_features_and_target(train_set), separate_features_and_target(test_set)

def optimize_hyperparameters(X_train, y_train, X_val, y_val, max_evals=50):
    """Optimize logistic regression hyperparameters using hyperopt"""
    
    space = {
        'C': hp.loguniform('C', np.log(0.001), np.log(100)),
        'penalty': hp.choice('penalty', ['l1', 'l2']),
        'solver': hp.choice('solver', ['liblinear', 'saga']),
        'max_iter': hp.choice('max_iter', [500, 1000, 2000])
    }
    
    def objective(params):
        with mlflow.start_run():
            # Handle solver-penalty compatibility
            if params['penalty'] == 'l1':
                params['solver'] = 'liblinear'
            
            model = LogisticRegression(random_state=42, **params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='weighted')
            mlflow.set_tag('Run type', 'Hyperparamater optimization')
            # Return negative F1 since hyperopt minimizes
            return {'loss': -f1, 'status': STATUS_OK}
    
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, 
                max_evals=max_evals, trials=trials)
    best_eval = space_eval(space, best)
    return best_eval, trials

def run_train():
    # Load combined dataset and separate data
    comb_df = load_data('data/processed/combined_spam_ham_dataset.pkl')
    (y_train, X_train), (y_test, X_test) = split_data(comb_df)
    
    # X_train and X_test are now properly formatted numpy arrays from separate_features_and_target
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Optimize hyperparameters
    print("Optimizing hyperparameters...")
    best_params, trials = optimize_hyperparameters(
        X_train_split, y_train_split, X_val, y_val, max_evals=50
    )
  
    
    # Train final model with best parameters

    print(f"Best parameters: {best_params}")
    with mlflow.start_run(run_name=f"optimized_model_{uuid.uuid4().hex[:4]}"):
        # Handle solver-penalty compatibility
        if best_params['penalty'] == 'l1':
            best_params['solver'] = 'liblinear'
            
        model = LogisticRegression(random_state=42, **best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mlflow.log_metric("test_f1_score", f1)
        mlflow.set_tag('Run type', 'test model training')

        print(f"Final F1 score: {f1}")
        run_id = mlflow.active_run().info.run_id
        print(f"Logged data and model in run {run_id}")

if __name__ == "__main__":
    run_train()




