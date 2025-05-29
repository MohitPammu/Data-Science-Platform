"""
XGBoost Healthcare Fraud Detection Model
Healthcare Intelligence Platform - Mohit Pammu Portfolio

Advanced gradient boosting implementation for healthcare fraud detection
with hyperparameter optimization and MLFlow integration.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Tuple, Any
import logging
from mlflow.models.signature import infer_signature

# Import our base model class
from src.models.base_healthcare_model import BaseHealthcareModel

class XGBoostFraudDetector(BaseHealthcareModel):
    """
    XGBoost model for healthcare fraud detection.
    
    Leverages gradient boosting for superior fraud pattern recognition
    with advanced hyperparameter tuning and interpretability features.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, random_state: int = 42, **kwargs):
        """
        Initialize XGBoost fraud detector.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Boosting learning rate
            random_state: Random seed for reproducibility
            **kwargs: Additional XGBClassifier parameters
        """
        super().__init__(model_name="XGBoost_Fraud", 
                        experiment_name="Healthcare_Fraud_Detection")
        
        # Initialize XGBoost model
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            scale_pos_weight=2,  # Handle imbalanced healthcare data
            eval_metric='logloss',  # Prevent warnings
            **kwargs
        )
        
        self.feature_importance_ = None
        self.shap_values = None
        self.best_params_ = None
        
        self.logger.info(f"Initialized XGBoost with {n_estimators} estimators, max_depth={max_depth}, lr={learning_rate}")
        
    def train_with_advanced_tuning(self, X: pd.DataFrame, y: pd.Series, 
                                  tune_hyperparameters: bool = True) -> Dict[str, float]:
        """
        Train XGBoost with advanced hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target variable  
            tune_hyperparameters: Whether to perform hyperparameter search
            
        Returns:
            Dictionary of training results and best parameters
        """
        self.logger.info("Training XGBoost for healthcare fraud detection...")
        
        with mlflow.start_run(run_name=f"XGBoost_Training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log initial parameters
            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_param("n_estimators", self.model.n_estimators)
            mlflow.log_param("max_depth", self.model.max_depth)
            mlflow.log_param("learning_rate", self.model.learning_rate)
            mlflow.log_param("scale_pos_weight", self.model.scale_pos_weight)
            mlflow.log_param("hyperparameter_tuning", tune_hyperparameters)
            
            if tune_hyperparameters:
                # Advanced hyperparameter search space for XGBoost
                param_dist = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6, 7],
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5],
                    'gamma': [0, 0.1, 0.2],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [1, 1.5, 2]
                }
                
                self.logger.info("Performing advanced XGBoost hyperparameter tuning...")
                
                # Randomized search with more iterations for XGBoost
                random_search = RandomizedSearchCV(
                    self.model, 
                    param_dist, 
                    n_iter=30,  # More iterations for XGBoost
                    cv=5,       # 5-fold CV for better estimates
                    scoring='f1',  # F1 score good for imbalanced data
                    random_state=42,
                    n_jobs=-1,
                    verbose=1
                )
                
                random_search.fit(X, y)
                
                # Update model with best parameters
                self.model = random_search.best_estimator_
                self.best_params_ = random_search.best_params_
                
                # Log best parameters
                for param, value in random_search.best_params_.items():
                    mlflow.log_param(f"best_{param}", value)
                    
                mlflow.log_metric("best_cv_f1_score", random_search.best_score_)
                
                self.logger.info(f"Best CV F1 Score: {random_search.best_score_:.3f}")
                self.logger.info(f"Best parameters: {random_search.best_params_}")
                
            else:
                # Train with default parameters
                self.model.fit(X, y)
                
            self.is_trained = True
            
            # Calculate feature importance (XGBoost style)
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
            self._create_xgboost_evaluation_plots(y, y_pred, y_pred_proba, X)
            
            # Calculate SHAP values for interpretability
            self._calculate_xgboost_shap_values(X.sample(min(100, len(X))))  # Sample for speed
            
            # Save model with signature and input example for production deployment
            input_example = X.head(3)  # Use first 3 rows as example
            signature = infer_signature(X, y_pred)
            
            # Save model artifact
            model_path = self.save_model()
            mlflow.log_artifact(model_path)
            mlflow.xgboost.log_model(
                xgb_model=self.model, 
                artifact_path="xgboost_model",
                signature=signature,
                input_example=input_example
            )
            
            training_results = {
                'model_trained': True,
                'feature_count': len(X.columns),
                'sample_count': len(X),
                'fraud_rate': y.sum() / len(y),
                'top_feature': self.feature_importance_.iloc[0]['feature'],
                'business_value': business_metrics['net_business_value'],
                'best_params': self.best_params_ if tune_hyperparameters else 'default'
            }
            
            self.logger.info("XGBoost training completed successfully!")
            return training_results
    
    def _create_xgboost_evaluation_plots(self, y_true: pd.Series, y_pred: np.ndarray, 
                                       y_pred_proba: np.ndarray, X: pd.DataFrame):
        """Create XGBoost-specific evaluation visualizations."""
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title('XGBoost Healthcare Fraud Detection - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/artifacts/xgb_confusion_matrix.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('models/artifacts/xgb_confusion_matrix.png')
        plt.close()
        
        # 2. Feature Importance Plot (XGBoost style)
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance_.head(15)
        sns.barplot(data=top_features, x='importance', y='feature', hue='feature', palette='viridis', legend=False)
        plt.title('Top 15 Features for XGBoost Healthcare Fraud Detection')
        plt.xlabel('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig('models/artifacts/xgb_feature_importance.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('models/artifacts/xgb_feature_importance.png')
        plt.close()
        
        # 3. ROC Curve (XGBoost performs well with probability calibration)
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('XGBoost ROC Curve')
        plt.legend(loc="lower right")
        
        plt.subplot(1, 2, 2)
        # Learning curve would go here if we had validation data
        plt.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Legitimate', color='blue')
        plt.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Fraud', color='red')
        plt.xlabel('Fraud Probability')
        plt.ylabel('Count')
        plt.title('XGBoost Prediction Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/artifacts/xgb_roc_and_distribution.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('models/artifacts/xgb_roc_and_distribution.png')
        plt.close()
        
    def _calculate_xgboost_shap_values(self, X_sample: pd.DataFrame):
        """Calculate SHAP values for XGBoost model interpretability."""
        try:
            self.logger.info("Calculating SHAP values for XGBoost interpretability...")
            
            # Create SHAP explainer for XGBoost
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            
            # Store SHAP values
            self.shap_values = shap_values
            
            # Create SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values, X_sample, plot_type="bar", show=False)
            plt.title('SHAP Feature Importance - XGBoost Healthcare Fraud Detection')
            plt.tight_layout()
            plt.savefig('models/artifacts/xgb_shap_summary.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact('models/artifacts/xgb_shap_summary.png')
            plt.close()
            
            self.logger.info("XGBoost SHAP analysis completed successfully")
            
        except Exception as e:
            self.logger.warning(f"XGBoost SHAP calculation failed: {e}")
            
    def get_feature_importance(self) -> pd.DataFrame:
        """Get XGBoost feature importance rankings."""
        if self.feature_importance_ is None:
            raise ValueError("Model must be trained first")
        return self.feature_importance_
    
    def predict_fraud_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability for new data using XGBoost.
        
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
        Explain a single prediction using XGBoost feature importance.
        
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
            'model_confidence': max(fraud_prob, 1 - fraud_prob),
            'model_type': 'XGBoost'
        }
        
        return explanation

def test_xgboost():
    """Test function for XGBoost fraud detection."""
    print("🚀 Testing XGBoost Fraud Detection...")
    
    # Load data directly from files
    providers_df = pd.read_csv('data/raw/medicare_provider_data.csv')
    claims_df = pd.read_csv('data/raw/medicare_claims_data.csv')
    print(f"✅ Loaded {len(providers_df)} providers and {len(claims_df)} claims from files")
    
    # Initialize XGBoost model
    xgb_detector = XGBoostFraudDetector(n_estimators=100, max_depth=6, learning_rate=0.1)
    print("✅ XGBoost model initialized")
    
    # Prepare features
    X, y = xgb_detector.prepare_healthcare_features(providers_df, claims_df)
    print(f"✅ Prepared {len(X.columns)} features for {len(X)} samples")
    print(f"Fraud rate: {y.mean():.1%}")
    
    # Train model (with hyperparameter tuning)
    print("\n🚀 Training XGBoost model with hyperparameter tuning...")
    results = xgb_detector.train_with_advanced_tuning(X, y, tune_hyperparameters=True)
    
    print("\n🎯 XGBoost Training Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Show top features
    print(f"\n🔍 Top 5 Important Features (XGBoost):")
    top_features = xgb_detector.get_feature_importance().head(5)
    for _, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Test prediction on sample
    sample_prediction = xgb_detector.explain_prediction(X.head(1))
    print(f"\n📊 Sample Prediction Explanation (XGBoost):")
    print(f"  Fraud Probability: {sample_prediction['fraud_probability']:.3f}")
    print(f"  Risk Level: {sample_prediction['risk_level']}")
    print(f"  Model Type: {sample_prediction['model_type']}")
    
    # Show business metrics
    business_metrics = xgb_detector.business_metrics
    print(f"\n💰 Business Impact (XGBoost):")
    print(f"  Potential Fraud Detected: {business_metrics['potential_fraud_detected']} cases")
    print(f"  Estimated Savings: ${business_metrics['estimated_savings']:,.2f}")
    print(f"  Investigation Costs: ${business_metrics['investigation_costs']:,.2f}")
    print(f"  Net Business Value: ${business_metrics['net_business_value']:,.2f}")
    
    print("\n✅ XGBoost test completed successfully!")
    print("🔬 Check MLFlow at http://localhost:5001 for detailed tracking!")

if __name__ == "__main__":
    test_xgboost()