"""
Base Healthcare Model Class
Healthcare Intelligence Platform - Mohit Pammu Portfolio

Production-ready base class for all healthcare ML models with MLFlow integration,
healthcare-specific preprocessing, and business metrics calculation.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
import joblib
from datetime import datetime
import os

# Import our existing Day 2 components
import sys
sys.path.append('.')
from src.feature_engineering.fraud_indicators import HealthcareFraudDetector
from src.data_ingestion.cms_connector import CMSDataConnector

class BaseHealthcareModel:
    """
    Base class for healthcare ML models with MLFlow integration and business metrics.
    
    Designed for production deployment with proper logging, error handling,
    and healthcare domain-specific evaluation metrics.
    """
    
    def __init__(self, model_name: str, experiment_name: str = "Healthcare_ML_Models"):
        """
        Initialize base healthcare model.
        
        Args:
            model_name: Name of the specific model (e.g., "RandomForest_Fraud")
            experiment_name: MLFlow experiment name
        """
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.model = None
        self.is_trained = False
        self.feature_names = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize MLFlow
        self._setup_mlflow()
        
        # Healthcare-specific business metrics
        self.business_metrics = {}
        
    def _setup_mlflow(self):
        """Set up MLFlow experiment tracking."""
        try:
            mlflow.set_tracking_uri("http://localhost:5001")
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(self.experiment_name)
                    self.logger.info(f"Created new MLFlow experiment: {self.experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    self.logger.info(f"Using existing MLFlow experiment: {self.experiment_name}")
                    
                mlflow.set_experiment(self.experiment_name)
                
            except Exception as e:
                self.logger.warning(f"MLFlow experiment setup issue: {e}")
                
        except Exception as e:
            self.logger.error(f"MLFlow connection failed: {e}")
            
    def prepare_healthcare_features(self, providers_df: pd.DataFrame, 
                                  claims_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features using our Day 2 fraud detection system.
        
        Args:
            providers_df: Medicare provider data
            claims_df: Medicare claims data
            
        Returns:
            Tuple of (features_df, target_series)
        """
        self.logger.info("Preparing healthcare features using Day 2 fraud detection system...")
        
        try:
            # Use our existing fraud detector with correct interface
            fraud_detector = HealthcareFraudDetector()
            
            # Get fraud analysis results using both methods
            billing_patterns = fraud_detector.analyze_provider_billing_patterns(providers_df, claims_df)
            fraud_risk_report = fraud_detector.generate_fraud_risk_report(providers_df, claims_df)
            
            # Combine results into unified fraud_results format
            fraud_results = {
                'billing_patterns': billing_patterns,
                'fraud_risk_report': fraud_risk_report,
                'total_providers': len(providers_df)
            }
            
            # Extract features and create target variable
            features_df = self._create_ml_features(fraud_results, providers_df, claims_df)
            target_series = self._create_fraud_labels(fraud_results, providers_df)
            
            self.feature_names = features_df.columns.tolist()
            self.logger.info(f"Created {len(features_df.columns)} features for {len(features_df)} providers")
            
            return features_df, target_series
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            raise
            
    def _create_ml_features(self, fraud_results: Dict, providers_df: pd.DataFrame, 
                      claims_df: pd.DataFrame) -> pd.DataFrame:
        """Create ML-ready features from fraud detection results."""
    
        # Base provider features
        features_df = providers_df.copy()
    
        # Add claim aggregations per provider
        claim_aggs = claims_df.groupby('provider_id').agg({
            'claim_amount': ['count', 'sum', 'mean', 'std'],
            'procedure_code': 'nunique',
            'patient_age': ['mean', 'std'],
            'service_count': ['sum', 'mean']
        }).round(2)
    
        # Flatten column names
        claim_aggs.columns = ['_'.join(col).strip() for col in claim_aggs.columns]
        claim_aggs = claim_aggs.fillna(0)
    
        # Merge provider and claim features
        features_df = features_df.merge(claim_aggs, left_on='provider_id', 
                                  right_index=True, how='left')
    
        # Add fraud detection insights as features
        if 'billing_patterns' in fraud_results:
            billing_patterns = fraud_results['billing_patterns']
            # Convert billing patterns to features (assuming it returns per-provider metrics)
            for metric_name, metric_value in billing_patterns.items():
                if isinstance(metric_value, (int, float)):
                    features_df[f'billing_{metric_name}'] = metric_value
    
        # Encode categorical variables
        categorical_cols = features_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'provider_id':  # Keep provider_id for reference
                features_df[f'{col}_encoded'] = pd.factorize(features_df[col])[0]
    
        # Select numerical features for ML
        feature_cols = features_df.select_dtypes(include=[np.number]).columns
        ml_features = features_df[feature_cols].fillna(0)
    
        # SCHEMA FIX: Convert all features to float64 to handle missing values in production
        ml_features = ml_features.astype('float64')
    
        self.logger.info(f"Features converted to float64 for production compatibility")
    
        return ml_features
        
    def _create_fraud_labels(self, fraud_results: Dict, providers_df: pd.DataFrame) -> pd.Series:
        """Create binary fraud labels from fraud detection results."""
        
        # Use provider payment amounts to create realistic fraud labels
        # Top 20% of providers by total_payment are considered high-risk
        payment_threshold = providers_df['total_payment'].quantile(0.8)
        
        # Also consider providers with unusually high claim counts
        claim_threshold = providers_df['total_claims'].quantile(0.85)
        
        # Create binary labels: high payment OR high claim count
        fraud_labels = (
            (providers_df['total_payment'] >= payment_threshold) |
            (providers_df['total_claims'] >= claim_threshold)
        ).astype(int)
        
        # Set index to provider_id for proper alignment
        labels = pd.Series(fraud_labels.values, 
                          index=providers_df['provider_id'], 
                          name='is_fraud')
        
        fraud_rate = labels.sum() / len(labels)
        self.logger.info(f"Created fraud labels with {fraud_rate:.1%} fraud rate")
        
        return labels
    
    def train_with_cross_validation(self, X: pd.DataFrame, y: pd.Series, 
                                  cv_folds: int = 5) -> Dict[str, float]:
        """
        Train model with cross-validation and MLFlow tracking.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation scores
        """
        if self.model is None:
            raise ValueError("Model not initialized. Subclass must set self.model")
            
        self.logger.info(f"Training {self.model_name} with {cv_folds}-fold cross-validation...")
        
        # Start MLFlow run
        with mlflow.start_run(run_name=f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_param("model_type", self.model_name)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples", X.shape[0])
            mlflow.log_param("cv_folds", cv_folds)
            
            # Cross-validation with stratified splits (important for imbalanced healthcare data)
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Calculate CV scores for multiple metrics
            cv_scores = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric)
                cv_scores[f'cv_{metric}_mean'] = scores.mean()
                cv_scores[f'cv_{metric}_std'] = scores.std()
                
                # Log to MLFlow
                mlflow.log_metric(f"cv_{metric}_mean", scores.mean())
                mlflow.log_metric(f"cv_{metric}_std", scores.std())
            
            # Train final model on full dataset
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate business metrics
            y_pred = self.model.predict(X)
            business_metrics = self._calculate_business_metrics(y, y_pred)
            
            # Log business metrics to MLFlow
            for metric, value in business_metrics.items():
                mlflow.log_metric(f"business_{metric}", value)
            
            # Save model artifact
            model_path = f"models/trained/{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            joblib.dump(self.model, model_path)
            mlflow.log_artifact(model_path)
            mlflow.sklearn.log_model(self.model, "model")
            
            self.logger.info(f"Model training complete. CV accuracy: {cv_scores['cv_accuracy_mean']:.3f}")
            
            return cv_scores
    
    def _calculate_business_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate healthcare-specific business metrics.
        
        These metrics translate technical performance into business value.
        """
        # Basic confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Healthcare business assumptions (based on DHCF experience)
        avg_claim_amount = 1500  # Average Medicare claim amount
        investigation_cost = 200  # Cost to investigate a flagged claim
        recovery_rate = 0.75     # Percentage of fraud cases where money is recovered
        
        # Business impact calculations
        business_metrics = {
            'potential_fraud_detected': tp,
            'false_alarms': fp,
            'missed_fraud_cases': fn,
            'investigation_efficiency': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'fraud_recovery_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'estimated_savings': tp * avg_claim_amount * recovery_rate,
            'investigation_costs': (tp + fp) * investigation_cost,
            'net_business_value': (tp * avg_claim_amount * recovery_rate) - ((tp + fp) * investigation_cost)
        }
        
        self.business_metrics = business_metrics
        return business_metrics
    
    def generate_evaluation_report(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Technical metrics
        report = {
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'business_metrics': self.business_metrics,
            'feature_importance': self.get_feature_importance() if hasattr(self, 'get_feature_importance') else None
        }
        
        return report
    
    def save_model(self, filepath: str = None):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        if filepath is None:
            filepath = f"models/trained/{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'business_metrics': self.business_metrics,
            'model_name': self.model_name
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
        return filepath