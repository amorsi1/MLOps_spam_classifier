import mlflow
import os
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import pickle 
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

mlflow.set_tracking_uri('sqlite:///mlflow') #configure backend DB
mlflow.set_experiment('embedding-spam-ham-classifier')

def load_data(combined_df_path: str):
# def combine_datasets(List[pd.DataFrame]):
    with open(combined_df_path, 'rb') as f:
        df = pickle.load(f)
    return df

def separate_features_and_target(df,target='label', features=['embedding']):
    return df[target], df[features]

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the DataFrame into training and testing sets.
    """
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
    return separate_features_and_target(train_set), separate_features_and_target(test_set)





