import mlflow
import mlflow.sklearn
import os
from datetime import datetime
import json

class MLFlowManager:
    def __init__(self, experiment_name: str = "healthcare-fraud-detection"):
        self.experiment_name = experiment_name
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: str = None):
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return mlflow.start_run(run_name=run_name)
    
    def log_model_performance(self, model, X_test, y_test, model_name: str):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
        
        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Log model
        mlflow.sklearn.log_model(model, model_name)
        
        return metrics
    
    def log_business_impact(self, cost_savings: float, efficiency_gain: float, roi: float):
        mlflow.log_metric("cost_savings_annual", cost_savings)
        mlflow.log_metric("efficiency_gain_percent", efficiency_gain)
        mlflow.log_metric("roi_ratio", roi)

# Global MLFlow manager instance
mlflow_manager = MLFlowManager()