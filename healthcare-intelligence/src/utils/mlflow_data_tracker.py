"""
MLFlow Data Tracking for Healthcare Intelligence
Tracks data quality, experiments, and preprocessing steps
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class HealthcareDataTracker:
    """Track healthcare data processing and quality metrics in MLFlow"""
    
    def __init__(self, experiment_name: str = "Healthcare-Data-Pipeline"):
        self.experiment_name = experiment_name
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Initialize MLFlow tracking"""
        # Set MLFlow tracking URI to our Docker container
        mlflow.set_tracking_uri("http://localhost:5001")
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new MLFlow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLFlow experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logger.error(f"MLFlow setup error: {e}")
            logger.info("Continuing without MLFlow tracking")
    
    def track_data_quality(self, df: pd.DataFrame, dataset_name: str, 
                          additional_metrics: Optional[Dict[str, Any]] = None):
        """Track data quality metrics in MLFlow"""
        with mlflow.start_run(run_name=f"data_quality_{dataset_name}"):
            # Basic data quality metrics
            metrics = {
                "total_records": len(df),
                "total_columns": len(df.columns),
                "missing_values_total": df.isnull().sum().sum(),
                "missing_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                "duplicate_records": df.duplicated().sum(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            # Add column-specific metrics
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    metrics[f"{col}_mean"] = df[col].mean()
                    metrics[f"{col}_std"] = df[col].std()
                    metrics[f"{col}_min"] = df[col].min()
                    metrics[f"{col}_max"] = df[col].max()
                    metrics[f"{col}_null_count"] = df[col].isnull().sum()
            
            # Add any additional metrics
            if additional_metrics:
                metrics.update(additional_metrics)
            
            # Log all metrics to MLFlow
            for key, value in metrics.items():
                if pd.notna(value):  # Only log non-null values
                    mlflow.log_metric(key, float(value))
            
            # Log dataset info as parameters
            mlflow.log_param("dataset_name", dataset_name)
            mlflow.log_param("data_shape", f"{df.shape[0]}x{df.shape[1]}")
            mlflow.log_param("columns", list(df.columns))
            
            logger.info(f"Tracked data quality for {dataset_name}: {len(df)} records, {len(df.columns)} columns")
            return metrics
    
    def track_preprocessing_step(self, step_name: str, input_shape: tuple, 
                               output_shape: tuple, parameters: Dict[str, Any]):
        """Track individual preprocessing steps"""
        with mlflow.start_run(run_name=f"preprocessing_{step_name}"):
            mlflow.log_param("preprocessing_step", step_name)
            mlflow.log_param("input_shape", f"{input_shape[0]}x{input_shape[1]}")
            mlflow.log_param("output_shape", f"{output_shape[0]}x{output_shape[1]}")
            
            # Log step-specific parameters
            for key, value in parameters.items():
                mlflow.log_param(key, value)
            
            # Log shape change metrics
            mlflow.log_metric("records_change", output_shape[0] - input_shape[0])
            mlflow.log_metric("columns_change", output_shape[1] - input_shape[1])
            mlflow.log_metric("data_retention_rate", output_shape[0] / input_shape[0])
            
            logger.info(f"Tracked preprocessing step: {step_name}")

    def track_fraud_indicators(self, df: pd.DataFrame, indicators: Dict[str, float]):
        """Track healthcare fraud detection indicators"""
        with mlflow.start_run(run_name="fraud_indicators_analysis"):
            # Log fraud indicators as metrics
            for indicator, value in indicators.items():
                mlflow.log_metric(f"fraud_indicator_{indicator}", value)
            
            # Log dataset context
            mlflow.log_param("analysis_type", "fraud_detection")
            mlflow.log_param("total_providers", len(df['provider_id'].unique()) if 'provider_id' in df.columns else 0)
            mlflow.log_param("analysis_date", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            logger.info("Tracked fraud indicators analysis")
