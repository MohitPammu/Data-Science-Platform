"""
Random Forest Healthcare Fraud Detection Model
Healthcare Intelligence Platform - Mohit Pammu Portfolio

Professional Random Forest implementation for healthcare fraud detection
with feature importance analysis and MLFlow integration.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Tuple, Any
import logging
from mlflow.models.signature import infer_signature

# Import our base model class
from src.models.base_healthcare_model import BaseHealthcareModel

class RandomForestFraudDetector(BaseHealthcareModel):
    """
    Random Forest model for healthcare fraud detection.
    
    Leverages ensemble learning for robust fraud pattern recognition
    with feature importance analysis for regulatory compliance.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
                 random_state: int = 42, **kwargs):
        """
        Initialize Random Forest fraud detector.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
            **kwargs: Additional RandomForestClassifier parameters
        """
        super().__init__(model_name="RandomForest_Fraud", 
                        experiment_name="Healthcare_Fraud_Detection")
        
        # Initialize Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',  # Handle imbalanced healthcare data
            **kwargs
        )
        
        self.feature_importance_ = None
        self.shap_values = None
        
        self.logger.info(f"Initialized Random Forest with {n_estimators} trees, max_depth={max_depth}")
        
    def train_with_hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                                        tune_hyperparameters: bool = True) -> Dict[str, float]:
        """
        Train Random Forest with optional hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target variable  
            tune_hyperparameters: Whether to perform hyperparameter search
            
        Returns:
            Dictionary of training results and best parameters
        """
        self.logger.info("Training Random Forest for healthcare fraud detection...")
        
        with mlflow.start_run(run_name=f"RandomForest_Training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log initial parameters
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", self.model.n_estimators)
            mlflow.log_param("max_depth", self.model.max_depth)
            mlflow.log_param("class_weight", self.model.class_weight)
            mlflow.log_param("hyperparameter_tuning", tune_hyperparameters)
            
            if tune_hyperparameters:
                # Hyperparameter search space
                param_dist = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
                
                self.logger.info("Performing hyperparameter tuning...")
                
                # Randomized search for efficiency
                random_search = RandomizedSearchCV(
                    self.model, 
                    param_dist, 
                    n_iter=20,  # Limited for speed
                    cv=3,       # 3-fold CV for healthcare data
                    scoring='f1',  # F1 score good for imbalanced data
                    random_state=42,
                    n_jobs=-1
                )
                
                random_search.fit(X, y)
                
                # Update model with best parameters
                self.model = random_search.best_estimator_
                
                # Log best parameters
                for param, value in random_search.best_params_.items():
                    mlflow.log_param(f"best_{param}", value)
                    
                mlflow.log_metric("best_cv_f1_score", random_search.best_score_)
                
                self.logger.info(f"Best CV F1 Score: {random_search.best_score_:.3f}")
                self.logger.info(f"Best parameters: {random_search.best_params_}")
                
            # Train the final model
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate feature importance
            self.feature_importance_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Log feature importance
            for idx, row in self.feature_importance_.head(10).iterrows():
                mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
            
            # Generate predictions for evaluation
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            
            # Calculate and log business metrics
            business_metrics = self._calculate_business_metrics(y, y_pred)
            for metric, value in business_metrics.items():
                mlflow.log_metric(f"business_{metric}", value)
            
            # Create and log visualizations
            self._create_evaluation_plots(y, y_pred, y_pred_proba, X)
            
            # Calculate SHAP values for interpretability
            self._calculate_shap_values(X.sample(min(100, len(X))))  # Sample for speed
            
            # Save model with signature and input example for production deployment
            input_example = X.head(3)  # Use first 3 rows as example
            signature = infer_signature(X, y_pred)
            
            # Save model artifact
            model_path = self.save_model()
            mlflow.log_artifact(model_path)
            mlflow.sklearn.log_model(
                sk_model=self.model, 
                artifact_path="random_forest_model",
                signature=signature,
                input_example=input_example
            )
            
            training_results = {
                'model_trained': True,
                'feature_count': len(X.columns),
                'sample_count': len(X),
                'fraud_rate': y.sum() / len(y),
                'top_feature': self.feature_importance_.iloc[0]['feature'],
                'business_value': business_metrics['net_business_value']
            }
            
            self.logger.info("Random Forest training completed successfully!")
            return training_results
    
    def _create_evaluation_plots(self, y_true: pd.Series, y_pred: np.ndarray, 
                               y_pred_proba: np.ndarray, X: pd.DataFrame):
        """Create evaluation visualizations."""
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title('Healthcare Fraud Detection - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/artifacts/rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('models/artifacts/rf_confusion_matrix.png')
        plt.close()
        
        # 2. Feature Importance Plot
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance_.head(15)
        sns.barplot(data=top_features, x='importance', y='feature', hue='feature', palette='viridis', legend=False)
        plt.title('Top 15 Features for Healthcare Fraud Detection')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig('models/artifacts/rf_feature_importance.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('models/artifacts/rf_feature_importance.png')
        plt.close()
        
        # 3. Prediction Distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Legitimate', color='blue')
        plt.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Fraud', color='red')
        plt.xlabel('Fraud Probability')
        plt.ylabel('Count')
        plt.title('Fraud Probability Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        fraud_rates_by_decile = []
        for i in range(10):
            decile_mask = (y_pred_proba >= i/10) & (y_pred_proba < (i+1)/10)
            if decile_mask.sum() > 0:
                fraud_rate = y_true[decile_mask].mean()
                fraud_rates_by_decile.append(fraud_rate)
            else:
                fraud_rates_by_decile.append(0)
        
        plt.bar(range(10), fraud_rates_by_decile, color='orange')
        plt.xlabel('Probability Decile')
        plt.ylabel('Actual Fraud Rate')
        plt.title('Model Calibration by Decile')
        plt.xticks(range(10), [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)], rotation=45)
        
        plt.tight_layout()
        plt.savefig('models/artifacts/rf_probability_analysis.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('models/artifacts/rf_probability_analysis.png')
        plt.close()
        
    def _calculate_shap_values(self, X_sample: pd.DataFrame):
        """Calculate SHAP values for model interpretability."""
        try:
            self.logger.info("Calculating SHAP values for model interpretability...")
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            
            # For binary classification, use positive class SHAP values
            if isinstance(shap_values, list):
                self.shap_values = shap_values[1]  # Fraud class
            else:
                self.shap_values = shap_values
            
            # Create SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values, X_sample, plot_type="bar", show=False)
            plt.title('SHAP Feature Importance - Healthcare Fraud Detection')
            plt.tight_layout()
            plt.savefig('models/artifacts/rf_shap_summary.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact('models/artifacts/rf_shap_summary.png')
            plt.close()
            
            self.logger.info("SHAP analysis completed successfully")
            
        except Exception as e:
            self.logger.warning(f"SHAP calculation failed: {e}")
            
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance rankings."""
        if self.feature_importance_ is None:
            raise ValueError("Model must be trained first")
        return self.feature_importance_
    
    def predict_fraud_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of fraud probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        return self.model.predict_proba(X)[:, 1]
    
    def explain_prediction(self, X_single: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.
        
        Args:
            X_single: Single row DataFrame
            
        Returns:
            Dictionary with prediction explanation
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        # Get prediction
        fraud_prob = self.predict_fraud_probability(X_single)[0]
        fraud_prediction = self.model.predict(X_single)[0]
        
        # Get top contributing features
        feature_contributions = []
        if self.feature_importance_ is not None:
            for _, row in self.feature_importance_.head(5).iterrows():
                feature_name = row['feature']
                if feature_name in X_single.columns:
                    feature_value = X_single[feature_name].iloc[0]
                    feature_contributions.append({
                        'feature': feature_name,
                        'value': feature_value,
                        'importance': row['importance']
                    })
        
        explanation = {
            'fraud_probability': fraud_prob,
            'predicted_fraud': bool(fraud_prediction),
            'risk_level': 'HIGH' if fraud_prob > 0.7 else 'MEDIUM' if fraud_prob > 0.3 else 'LOW',
            'top_contributing_features': feature_contributions,
            'model_confidence': max(fraud_prob, 1 - fraud_prob)
        }
        
        return explanation

def test_random_forest():
    """Test function for Random Forest fraud detection."""
    print("🌲 Testing Random Forest Fraud Detection...")
    
    # Import data components
    from src.data_ingestion.cms_connector import CMSDataConnector
    
    # Load data
    connector = CMSDataConnector()
    providers_df, claims_df = connector.load_data()
    
    print(f"Loaded {len(providers_df)} providers and {len(claims_df)} claims")
    
    # Initialize model
    rf_detector = RandomForestFraudDetector(n_estimators=50, max_depth=10)
    
    # Prepare features
    X, y = rf_detector.prepare_healthcare_features(providers_df, claims_df)
    print(f"Prepared {len(X.columns)} features for {len(X)} samples")
    print(f"Fraud rate: {y.mean():.1%}")
    
    # Train model
    results = rf_detector.train_with_hyperparameter_tuning(X, y, tune_hyperparameters=False)
    
    print("\n🎯 Training Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Show top features
    print(f"\n🔍 Top 5 Important Features:")
    top_features = rf_detector.get_feature_importance().head(5)
    for _, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Test prediction on sample
    sample_prediction = rf_detector.explain_prediction(X.head(1))
    print(f"\n📊 Sample Prediction Explanation:")
    print(f"  Fraud Probability: {sample_prediction['fraud_probability']:.3f}")
    print(f"  Risk Level: {sample_prediction['risk_level']}")
    
    print("✅ Random Forest test completed successfully!")

if __name__ == "__main__":
    test_random_forest()