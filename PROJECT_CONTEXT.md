# DATA SCIENCE PLATFORM - PROJECT CONTEXT

## Current Status: Day 2 Complete, Starting Day 3

## Architecture Overview:
- Healthcare Intelligence Platform with fraud detection
- MLFlow experiment tracking at localhost:5001  
- Shared infrastructure for 3-project portfolio
- Production-ready modules and error handling

## Key Components Built:
- CMS Data Connector: healthcare-intelligence/src/data_ingestion/cms_connector.py
- Fraud Detection: healthcare-intelligence/src/feature_engineering/fraud_indicators.py
- Geographic Analysis: healthcare-intelligence/src/feature_engineering/geographic_simple.py
- MLFlow Tracking: healthcare-intelligence/src/utils/mlflow_data_tracker.py

## Day 3 Goals:
- Random Forest and XGBoost classification models
- Isolation Forest anomaly detection  
- Model evaluation and comparison in MLFlow
- SHAP interpretability analysis

## Development Patterns:
- Always use MLFlow for experiment tracking
- Maintain modular architecture for reusability
- Follow healthcare compliance considerations
- Professional error handling and logging

## Virtual Environment:
- Activate: source data-science-env/bin/activate
- Key packages: pandas, numpy, scikit-learn, mlflow, plotly

## Testing Commands:
- MLFlow health: curl -s http://localhost:5001/health
- Data files: ls -la healthcare-intelligence/data/raw/
- Modules: python -c "import sys; sys.path.append('.'); from healthcare_intelligence.src... import..."
