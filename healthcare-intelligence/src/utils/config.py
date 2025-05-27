import os
from pathlib import Path
from typing import Dict, Any

class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # MLFlow configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    EXPERIMENT_NAME = "healthcare-fraud-detection"
    
    # API configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Business impact parameters
    AVERAGE_FRAUD_AMOUNT = 15000  # Average fraudulent claim amount
    MANUAL_REVIEW_COST = 50      # Cost per manual review
    INVESTIGATION_COST = 500     # Cost per fraud investigation
    
    # Model thresholds
    FRAUD_THRESHOLD = 0.7        # Probability threshold for fraud classification
    HIGH_RISK_THRESHOLD = 0.8    # Threshold for high-risk cases
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.MODELS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

config = Config()