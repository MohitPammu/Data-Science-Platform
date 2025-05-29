"""
Day 3 Quick Start Template
Use this if you need to start Day 3 independently
"""

# Essential imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
import mlflow
import mlflow.sklearn

# Load your Day 2 data
providers_df = pd.read_csv('data/raw/medicare_provider_data.csv')
claims_df = pd.read_csv('data/raw/medicare_claims_data.csv')

# MLFlow setup
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Healthcare-ML-Models")

# Basic model template
with mlflow.start_run(run_name="random_forest_fraud"):
    # Your model code here
    mlflow.log_param("model_type", "RandomForest")
    # mlflow.log_metric("accuracy", accuracy_score)
    pass

print("✅ Day 3 template ready!")
