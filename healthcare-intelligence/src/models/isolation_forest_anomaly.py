"""
Isolation Forest Healthcare Anomaly Detection Model
Healthcare Intelligence Platform - Mohit Pammu Portfolio

Unsupervised anomaly detection for healthcare fraud using Isolation Forest
with contamination tuning and MLFlow integration.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
from mlflow.models.signature import infer_signature

# Import our base model class
from src.models.base_healthcare_model import BaseHealthcareModel

class IsolationForestAnomalyDetector(BaseHealthcareModel):
    """
    Isolation Forest model for healthcare fraud anomaly detection.
    
    Unsupervised learning approach that identifies fraud patterns without 
    requiring labeled training data - valuable for discovering unknown fraud schemes.
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100, 
                 max_samples: str = 'auto', random_state: int = 42, **kwargs):
        """
        Initialize Isolation Forest anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers in the data
            n_estimators: Number of base estimators in the ensemble
            max_samples: Number of samples to draw from X to train each base estimator
            random_state: Random seed for reproducibility
            **kwargs: Additional IsolationForest parameters
        """
        super().__init__(model_name="IsolationForest_Anomaly", 
                        experiment_name="Healthcare_Fraud_Detection")
        
        # Initialize Isolation Forest model
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            **kwargs
        )
        
        self.anomaly_scores_ = None
        self.contamination_analysis_ = None
        self.optimal_contamination_ = None
        
        self.logger.info(f"Initialized Isolation Forest with contamination={contamination}, n_estimators={n_estimators}")
        
    def train_with_contamination_tuning(self, X: pd.DataFrame, y: pd.Series = None, 
                                      tune_contamination: bool = True) -> Dict[str, float]:
        """
        Train Isolation Forest with contamination rate tuning.
        
        Args:
            X: Feature matrix
            y: Target variable (optional - used for evaluation only)
            tune_contamination: Whether to perform contamination tuning
            
        Returns:
            Dictionary of training results
        """
        self.logger.info("Training Isolation Forest for healthcare anomaly detection...")
        
        with mlflow.start_run(run_name=f"IsolationForest_Training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log initial parameters
            mlflow.log_param("model_type", "IsolationForest")
            mlflow.log_param("n_estimators", self.model.n_estimators)
            mlflow.log_param("max_samples", self.model.max_samples)
            mlflow.log_param("contamination_tuning", tune_contamination)
            
            if tune_contamination and y is not None:
                # Test different contamination rates to find optimal
                contamination_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
                contamination_results = []
                
                self.logger.info("Tuning contamination rate for optimal anomaly detection...")
                
                for contamination in contamination_rates:
                    # Create temporary model with this contamination rate
                    temp_model = IsolationForest(
                        contamination=contamination,
                        n_estimators=self.model.n_estimators,
                        max_samples=self.model.max_samples,
                        random_state=42
                    )
                    
                    # Fit and predict
                    temp_model.fit(X)
                    predictions = temp_model.predict(X)
                    
                    # Convert to binary (1 for normal, -1 for anomaly)
                    # Convert to our format (0 for normal, 1 for anomaly)
                    anomaly_labels = (predictions == -1).astype(int)
                    
                    # Calculate metrics if we have true labels
                    precision = precision_score(y, anomaly_labels, zero_division=0)
                    recall = recall_score(y, anomaly_labels, zero_division=0)
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    contamination_results.append({
                        'contamination': contamination,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'anomalies_detected': anomaly_labels.sum()
                    })
                    
                    self.logger.info(f"Contamination {contamination:.2f}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
                
                # Find optimal contamination rate
                self.contamination_analysis_ = pd.DataFrame(contamination_results)
                best_result = self.contamination_analysis_.loc[self.contamination_analysis_['f1_score'].idxmax()]
                self.optimal_contamination_ = best_result['contamination']
                
                # Update model with optimal contamination
                self.model.contamination = self.optimal_contamination_
                
                # Log contamination tuning results
                mlflow.log_metric("optimal_contamination", self.optimal_contamination_)
                mlflow.log_metric("best_f1_score", best_result['f1_score'])
                
                self.logger.info(f"Optimal contamination rate: {self.optimal_contamination_:.3f}")
                
            # Train final model
            self.model.fit(X)
            self.is_trained = True
            
            # Generate anomaly scores and predictions
            self.anomaly_scores_ = self.model.decision_function(X)
            anomaly_predictions = self.model.predict(X)
            
            # Convert predictions to binary format (0 = normal, 1 = anomaly)
            binary_predictions = (anomaly_predictions == -1).astype(int)
            
            # Calculate business metrics if we have true labels
            if y is not None:
                business_metrics = self._calculate_business_metrics(y, binary_predictions)
                for metric, value in business_metrics.items():
                    mlflow.log_metric(f"business_{metric}", value)
            else:
                # Create mock business metrics for unsupervised case
                anomaly_count = binary_predictions.sum()
                business_metrics = {
                    'potential_fraud_detected': anomaly_count,
                    'anomaly_rate': anomaly_count / len(binary_predictions),
                    'estimated_savings': anomaly_count * 1500 * 0.75,  # Assume same recovery rate
                    'investigation_costs': anomaly_count * 200,
                    'net_business_value': (anomaly_count * 1500 * 0.75) - (anomaly_count * 200)
                }
                self.business_metrics = business_metrics
                
                for metric, value in business_metrics.items():
                    mlflow.log_metric(f"unsupervised_{metric}", value)
            
            # Create and log visualizations
            self._create_anomaly_evaluation_plots(X, binary_predictions, y)
            
            # Save model with signature and input example
            input_example = X.head(3)
            signature = infer_signature(X, binary_predictions)
            
            # Save model artifact
            model_path = self.save_model()
            mlflow.log_artifact(model_path)
            mlflow.sklearn.log_model(
                sk_model=self.model, 
                artifact_path="isolation_forest_model",
                signature=signature,
                input_example=input_example
            )
            
            training_results = {
                'model_trained': True,
                'feature_count': len(X.columns),
                'sample_count': len(X),
                'contamination_rate': self.model.contamination,
                'anomalies_detected': binary_predictions.sum(),
                'optimal_contamination': self.optimal_contamination_ if tune_contamination else None,
                'business_value': business_metrics.get('net_business_value', business_metrics.get('estimated_savings', 0))
            }
            
            self.logger.info("Isolation Forest training completed successfully!")
            return training_results
    
    def _create_anomaly_evaluation_plots(self, X: pd.DataFrame, predictions: np.ndarray, y: pd.Series = None):
        """Create anomaly detection specific visualizations."""
        
        # 1. Anomaly Score Distribution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.hist(self.anomaly_scores_, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='Decision Boundary')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        
        # 2. Anomaly Detection Results
        plt.subplot(2, 3, 2)
        anomaly_counts = pd.Series(predictions).value_counts()
        plt.pie([anomaly_counts.get(0, 0), anomaly_counts.get(1, 0)], 
                labels=['Normal', 'Anomaly'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        plt.title('Anomaly Detection Results')
        
        # 3. Contamination Tuning Results (if available)
        if self.contamination_analysis_ is not None:
            plt.subplot(2, 3, 3)
            plt.plot(self.contamination_analysis_['contamination'], 
                    self.contamination_analysis_['f1_score'], 'bo-', label='F1 Score')
            plt.plot(self.contamination_analysis_['contamination'], 
                    self.contamination_analysis_['precision'], 'ro-', label='Precision')
            plt.plot(self.contamination_analysis_['contamination'], 
                    self.contamination_analysis_['recall'], 'go-', label='Recall')
            plt.xlabel('Contamination Rate')
            plt.ylabel('Score')
            plt.title('Contamination Rate Optimization')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. Feature vs Anomaly Score (top 2 important features)
        if len(X.columns) >= 2:
            plt.subplot(2, 3, 4)
            # Use the first two numerical features for visualization
            feature1 = X.columns[0]
            feature2 = X.columns[1]
            
            scatter = plt.scatter(X[feature1], X[feature2], c=self.anomaly_scores_, 
                                cmap='RdYlBu_r', alpha=0.6)
            plt.colorbar(scatter, label='Anomaly Score')
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.title('Feature Space with Anomaly Scores')
        
        # 5. Confusion Matrix (if true labels available)
        if y is not None:
            plt.subplot(2, 3, 5)
            cm = confusion_matrix(y, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'])
            plt.title('Anomaly Detection Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        # 6. Anomaly Score vs Feature Importance
        plt.subplot(2, 3, 6)
        # Calculate correlation between each feature and anomaly scores
        feature_correlations = []
        for col in X.columns:
            corr = np.corrcoef(X[col], self.anomaly_scores_)[0, 1]
            if not np.isnan(corr):
                feature_correlations.append((col, abs(corr)))
        
        if feature_correlations:
            feature_correlations.sort(key=lambda x: x[1], reverse=True)
            top_features = feature_correlations[:10]
            
            features, correlations = zip(*top_features)
            plt.barh(range(len(features)), correlations, color='orange', alpha=0.7)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Absolute Correlation with Anomaly Score')
            plt.title('Feature Relevance for Anomaly Detection')
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('models/artifacts/isolation_forest_analysis.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('models/artifacts/isolation_forest_analysis.png')
        plt.close()
        
    def predict_anomalies(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (binary_predictions, anomaly_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        anomaly_scores = self.model.decision_function(X)
        predictions = self.model.predict(X)
        binary_predictions = (predictions == -1).astype(int)
        
        return binary_predictions, anomaly_scores
    
    def explain_anomaly(self, X_single: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain why a single instance was flagged as an anomaly.
        
        Args:
            X_single: Single row DataFrame
            
        Returns:
            Dictionary with anomaly explanation
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        # Get prediction and score
        binary_pred, anomaly_score = self.predict_anomalies(X_single)
        
        # Calculate feature contributions (simplified approach)
        feature_contributions = []
        for col in X_single.columns:
            value = X_single[col].iloc[0]
            # Simple approach: features further from median contribute more to anomaly
            if pd.api.types.is_numeric_dtype(X_single[col]):
                feature_contributions.append({
                    'feature': col,
                    'value': value,
                    'contribution': 'feature_analysis'  # Placeholder for more complex analysis
                })
        
        explanation = {
            'anomaly_probability': float(binary_pred[0]),
            'anomaly_score': float(anomaly_score[0]),
            'is_anomaly': bool(binary_pred[0]),
            'risk_level': 'HIGH' if binary_pred[0] else 'NORMAL',
            'top_contributing_features': feature_contributions[:5],
            'model_type': 'IsolationForest',
            'detection_method': 'Unsupervised Anomaly Detection'
        }
        
        return explanation

def test_isolation_forest():
    """Test function for Isolation Forest anomaly detection."""
    print("🌳 Testing Isolation Forest Anomaly Detection...")
    
    # Load data directly from files
    providers_df = pd.read_csv('data/raw/medicare_provider_data.csv')
    claims_df = pd.read_csv('data/raw/medicare_claims_data.csv')
    print(f"✅ Loaded {len(providers_df)} providers and {len(claims_df)} claims from files")
    
    # Initialize Isolation Forest model
    if_detector = IsolationForestAnomalyDetector(contamination=0.334, n_estimators=100)  # Match our fraud rate
    print("✅ Isolation Forest model initialized")
    
    # Prepare features
    X, y = if_detector.prepare_healthcare_features(providers_df, claims_df)
    print(f"✅ Prepared {len(X.columns)} features for {len(X)} samples")
    print(f"True fraud rate: {y.mean():.1%}")
    
    # Train model (with contamination tuning)
    print("\n🌳 Training Isolation Forest with contamination tuning...")
    results = if_detector.train_with_contamination_tuning(X, y, tune_contamination=True)
    
    print("\n🎯 Isolation Forest Training Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Test prediction on sample
    sample_prediction = if_detector.explain_anomaly(X.head(1))
    print(f"\n📊 Sample Anomaly Explanation:")
    print(f"  Anomaly Score: {sample_prediction['anomaly_score']:.3f}")
    print(f"  Is Anomaly: {sample_prediction['is_anomaly']}")
    print(f"  Risk Level: {sample_prediction['risk_level']}")
    print(f"  Detection Method: {sample_prediction['detection_method']}")
    
    # Show business metrics - CORRECTED
    business_metrics = if_detector.business_metrics
    print(f"\n💰 Business Impact (Isolation Forest):")
    print(f"  Potential Fraud Detected: {business_metrics['potential_fraud_detected']} cases")
    print(f"  False Alarms: {business_metrics['false_alarms']} cases")
    print(f"  Investigation Efficiency: {business_metrics['investigation_efficiency']:.1%}")
    print(f"  Fraud Recovery Rate: {business_metrics['fraud_recovery_rate']:.1%}")
    print(f"  Estimated Savings: ${business_metrics['estimated_savings']:,.2f}")
    print(f"  Investigation Costs: ${business_metrics['investigation_costs']:,.2f}")
    print(f"  Net Business Value: ${business_metrics['net_business_value']:,.2f}")
    
    print("\n✅ Isolation Forest test completed successfully!")
    print("🔬 Check MLFlow at http://localhost:5001 for detailed tracking!")

if __name__ == "__main__":
    test_isolation_forest()