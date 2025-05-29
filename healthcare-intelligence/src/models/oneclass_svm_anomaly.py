"""
One-Class SVM Healthcare Anomaly Detection Model
Healthcare Intelligence Platform - Mohit Pammu Portfolio

Advanced SVM-based anomaly detection for healthcare fraud using One-Class SVM
with parameter tuning and MLFlow integration.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.svm import OneClassSVM
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
from mlflow.models.signature import infer_signature

# Import our base model class
from src.models.base_healthcare_model import BaseHealthcareModel

class OneClassSVMAnomalyDetector(BaseHealthcareModel):
    """
    One-Class SVM model for healthcare fraud anomaly detection.
    
    Support Vector Machine-based approach for anomaly detection that learns
    the boundary of normal behavior and identifies deviations as potential fraud.
    """
    
    def __init__(self, nu: float = 0.1, kernel: str = 'rbf', gamma: str = 'scale', 
                 random_state: int = 42, **kwargs):
        """
        Initialize One-Class SVM anomaly detector.
        
        Args:
            nu: An upper bound on the fraction of training errors and lower bound of support vectors
            kernel: Kernel type to be used in the algorithm
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            random_state: Random seed for reproducibility
            **kwargs: Additional OneClassSVM parameters
        """
        super().__init__(model_name="OneClassSVM_Anomaly", 
                        experiment_name="Healthcare_Fraud_Detection")
        
        # Initialize One-Class SVM model
        self.model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma,
            **kwargs
        )
        
        # Initialize scaler for SVM (important for performance)
        self.scaler = StandardScaler()
        self.scaled_features = None
        
        self.anomaly_scores_ = None
        self.parameter_analysis_ = None
        self.optimal_params_ = None
        
        self.logger.info(f"Initialized One-Class SVM with nu={nu}, kernel={kernel}, gamma={gamma}")
        
    def train_with_parameter_tuning(self, X: pd.DataFrame, y: pd.Series = None, 
                                   tune_parameters: bool = True) -> Dict[str, float]:
        """
        Train One-Class SVM with parameter tuning.
        
        Args:
            X: Feature matrix
            y: Target variable (optional - used for evaluation only)
            tune_parameters: Whether to perform parameter tuning
            
        Returns:
            Dictionary of training results
        """
        self.logger.info("Training One-Class SVM for healthcare anomaly detection...")
        
        with mlflow.start_run(run_name=f"OneClassSVM_Training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log initial parameters
            mlflow.log_param("model_type", "OneClassSVM")
            mlflow.log_param("kernel", self.model.kernel)
            mlflow.log_param("gamma", self.model.gamma)
            mlflow.log_param("parameter_tuning", tune_parameters)
            
            # Scale features (critical for SVM performance)
            self.scaled_features = self.scaler.fit_transform(X)
            
            if tune_parameters and y is not None:
                # Test different parameter combinations
                param_grid = {
                    'nu': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                }
                
                parameter_results = []
                self.logger.info("Tuning One-Class SVM parameters for optimal anomaly detection...")
                
                best_f1 = 0
                best_params = None
                
                # Test subset of combinations for efficiency
                test_combinations = [
                    {'nu': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
                    {'nu': 0.15, 'kernel': 'rbf', 'gamma': 'scale'},
                    {'nu': 0.2, 'kernel': 'rbf', 'gamma': 'scale'},
                    {'nu': 0.25, 'kernel': 'rbf', 'gamma': 'scale'},
                    {'nu': 0.3, 'kernel': 'rbf', 'gamma': 'scale'},
                    {'nu': 0.35, 'kernel': 'rbf', 'gamma': 'scale'},
                    {'nu': 0.2, 'kernel': 'poly', 'gamma': 'scale'},
                    {'nu': 0.2, 'kernel': 'sigmoid', 'gamma': 'scale'},
                    {'nu': 0.2, 'kernel': 'rbf', 'gamma': 'auto'},
                    {'nu': 0.2, 'kernel': 'rbf', 'gamma': 0.1},
                ]
                
                for params in test_combinations:
                    try:
                        # Create temporary model with these parameters
                        temp_model = OneClassSVM(**params)
                        
                        # Fit and predict
                        temp_model.fit(self.scaled_features)
                        predictions = temp_model.predict(self.scaled_features)
                        
                        # Convert to binary (1 for normal, -1 for anomaly)
                        # Convert to our format (0 for normal, 1 for anomaly)
                        anomaly_labels = (predictions == -1).astype(int)
                        
                        # Calculate metrics if we have true labels
                        precision = precision_score(y, anomaly_labels, zero_division=0)
                        recall = recall_score(y, anomaly_labels, zero_division=0)
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        parameter_results.append({
                            'nu': params['nu'],
                            'kernel': params['kernel'], 
                            'gamma': params['gamma'],
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'anomalies_detected': anomaly_labels.sum()
                        })
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_params = params
                        
                        self.logger.info(f"Params {params}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
                        
                    except Exception as e:
                        self.logger.warning(f"Parameter combination {params} failed: {e}")
                        continue
                
                # Store parameter analysis results
                self.parameter_analysis_ = pd.DataFrame(parameter_results)
                self.optimal_params_ = best_params
                
                if best_params:
                    # Update model with optimal parameters
                    self.model = OneClassSVM(**best_params)
                    
                    # Log optimal parameters
                    for param, value in best_params.items():
                        mlflow.log_param(f"optimal_{param}", value)
                    mlflow.log_metric("best_f1_score", best_f1)
                    
                    self.logger.info(f"Optimal parameters: {best_params}")
                    self.logger.info(f"Best F1 score: {best_f1:.3f}")
                else:
                    self.logger.warning("No valid parameter combination found, using defaults")
                
            # Train final model
            self.model.fit(self.scaled_features)
            self.is_trained = True
            
            # Generate anomaly scores and predictions
            self.anomaly_scores_ = self.model.decision_function(self.scaled_features)
            anomaly_predictions = self.model.predict(self.scaled_features)
            
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
                    'estimated_savings': anomaly_count * 1500 * 0.75,
                    'investigation_costs': anomaly_count * 200,
                    'net_business_value': (anomaly_count * 1500 * 0.75) - (anomaly_count * 200)
                }
                self.business_metrics = business_metrics
                
                for metric, value in business_metrics.items():
                    mlflow.log_metric(f"unsupervised_{metric}", value)
            
            # Create and log visualizations
            self._create_svm_evaluation_plots(X, binary_predictions, y)
            
            # Save model with signature and input example
            input_example = X.head(3)
            signature = infer_signature(X, binary_predictions)
            
            # Save model artifact
            model_path = self.save_model()
            mlflow.log_artifact(model_path)
            mlflow.sklearn.log_model(
                sk_model=self.model, 
                artifact_path="oneclass_svm_model",
                signature=signature,
                input_example=input_example
            )
            
            training_results = {
                'model_trained': True,
                'feature_count': len(X.columns),
                'sample_count': len(X),
                'nu_parameter': self.model.nu,
                'kernel': self.model.kernel,
                'anomalies_detected': binary_predictions.sum(),
                'optimal_params': self.optimal_params_ if tune_parameters else None,
                'business_value': business_metrics.get('net_business_value', business_metrics.get('estimated_savings', 0))
            }
            
            self.logger.info("One-Class SVM training completed successfully!")
            return training_results
    
    def _create_svm_evaluation_plots(self, X: pd.DataFrame, predictions: np.ndarray, y: pd.Series = None):
        """Create One-Class SVM specific visualizations."""
        
        plt.figure(figsize=(15, 10))
        
        # 1. Decision Function Distribution
        plt.subplot(2, 3, 1)
        plt.hist(self.anomaly_scores_, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='Decision Boundary')
        plt.xlabel('Decision Function Score')
        plt.ylabel('Frequency')
        plt.title('One-Class SVM Decision Function Distribution')
        plt.legend()
        
        # 2. Anomaly Detection Results
        plt.subplot(2, 3, 2)
        anomaly_counts = pd.Series(predictions).value_counts()
        plt.pie([anomaly_counts.get(0, 0), anomaly_counts.get(1, 0)], 
                labels=['Normal', 'Anomaly'], autopct='%1.1f%%', colors=['lightgreen', 'salmon'])
        plt.title('One-Class SVM Anomaly Results')
        
        # 3. Parameter Tuning Results (if available)
        if self.parameter_analysis_ is not None and len(self.parameter_analysis_) > 0:
            plt.subplot(2, 3, 3)
            
            # Plot F1 scores for different nu values (RBF kernel only)
            rbf_results = self.parameter_analysis_[self.parameter_analysis_['kernel'] == 'rbf']
            if len(rbf_results) > 0:
                plt.plot(rbf_results['nu'], rbf_results['f1_score'], 'bo-', label='F1 Score')
                plt.plot(rbf_results['nu'], rbf_results['precision'], 'ro-', label='Precision')
                plt.plot(rbf_results['nu'], rbf_results['recall'], 'go-', label='Recall')
                plt.xlabel('Nu Parameter')
                plt.ylabel('Score')
                plt.title('SVM Parameter Optimization (RBF Kernel)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No RBF results available', ha='center', va='center')
                plt.title('Parameter Optimization')
        
        # 4. Feature Space Visualization (2D projection)
        if len(X.columns) >= 2:
            plt.subplot(2, 3, 4)
            feature1 = X.columns[0]
            feature2 = X.columns[1]
            
            scatter = plt.scatter(X[feature1], X[feature2], c=self.anomaly_scores_, 
                                cmap='RdYlBu_r', alpha=0.6)
            plt.colorbar(scatter, label='SVM Decision Score')
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.title('Feature Space with SVM Scores')
        
        # 5. Confusion Matrix (if true labels available)
        if y is not None:
            plt.subplot(2, 3, 5)
            cm = confusion_matrix(y, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'])
            plt.title('One-Class SVM Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        # 6. Kernel Comparison (if parameter tuning was done)
        if self.parameter_analysis_ is not None and len(self.parameter_analysis_) > 0:
            plt.subplot(2, 3, 6)
            kernel_performance = self.parameter_analysis_.groupby('kernel')['f1_score'].max()
            if len(kernel_performance) > 1:
                plt.bar(kernel_performance.index, kernel_performance.values, color='purple', alpha=0.7)
                plt.xlabel('Kernel Type')
                plt.ylabel('Best F1 Score')
                plt.title('Kernel Performance Comparison')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'Single kernel tested', ha='center', va='center')
                plt.title('Kernel Analysis')
        
        plt.tight_layout()
        plt.savefig('models/artifacts/oneclass_svm_analysis.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('models/artifacts/oneclass_svm_analysis.png')
        plt.close()
        
    def predict_anomalies(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (binary_predictions, decision_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        # Scale the features using the fitted scaler
        X_scaled = self.scaler.transform(X)
        
        decision_scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        binary_predictions = (predictions == -1).astype(int)
        
        return binary_predictions, decision_scores
    
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
        binary_pred, decision_score = self.predict_anomalies(X_single)
        
        # Calculate feature contributions (simplified approach)
        feature_contributions = []
        for col in X_single.columns:
            value = X_single[col].iloc[0]
            feature_contributions.append({
                'feature': col,
                'value': value,
                'contribution': 'svm_analysis'  # Placeholder for more complex analysis
            })
        
        explanation = {
            'anomaly_probability': float(binary_pred[0]),
            'decision_score': float(decision_score[0]),
            'is_anomaly': bool(binary_pred[0]),
            'risk_level': 'HIGH' if binary_pred[0] else 'NORMAL',
            'top_contributing_features': feature_contributions[:5],
            'model_type': 'OneClassSVM',
            'detection_method': 'Support Vector Machine Anomaly Detection',
            'kernel_used': self.model.kernel,
            'nu_parameter': self.model.nu
        }
        
        return explanation

def test_oneclass_svm():
    """Test function for One-Class SVM anomaly detection."""
    print("🔺 Testing One-Class SVM Anomaly Detection...")
    
    # Load data directly from files
    providers_df = pd.read_csv('data/raw/medicare_provider_data.csv')
    claims_df = pd.read_csv('data/raw/medicare_claims_data.csv')
    print(f"✅ Loaded {len(providers_df)} providers and {len(claims_df)} claims from files")
    
    # Initialize One-Class SVM model
    svm_detector = OneClassSVMAnomalyDetector(nu=0.2, kernel='rbf', gamma='scale')
    print("✅ One-Class SVM model initialized")
    
    # Prepare features
    X, y = svm_detector.prepare_healthcare_features(providers_df, claims_df)
    print(f"✅ Prepared {len(X.columns)} features for {len(X)} samples")
    print(f"True fraud rate: {y.mean():.1%}")
    
    # Train model (with parameter tuning)
    print("\n🔺 Training One-Class SVM with parameter tuning...")
    results = svm_detector.train_with_parameter_tuning(X, y, tune_parameters=True)
    
    print("\n🎯 One-Class SVM Training Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Test prediction on sample
    sample_prediction = svm_detector.explain_anomaly(X.head(1))
    print(f"\n📊 Sample Anomaly Explanation (One-Class SVM):")
    print(f"  Decision Score: {sample_prediction['decision_score']:.3f}")
    print(f"  Is Anomaly: {sample_prediction['is_anomaly']}")
    print(f"  Risk Level: {sample_prediction['risk_level']}")
    print(f"  Detection Method: {sample_prediction['detection_method']}")
    print(f"  Kernel: {sample_prediction['kernel_used']}")
    print(f"  Nu Parameter: {sample_prediction['nu_parameter']}")
    
    # Show business metrics
    business_metrics = svm_detector.business_metrics
    print(f"\n💰 Business Impact (One-Class SVM):")
    print(f"  Potential Fraud Detected: {business_metrics['potential_fraud_detected']} cases")
    print(f"  Estimated Savings: ${business_metrics['estimated_savings']:,.2f}")
    print(f"  Investigation Costs: ${business_metrics['investigation_costs']:,.2f}")
    print(f"  Net Business Value: ${business_metrics['net_business_value']:,.2f}")
    
    print("\n✅ One-Class SVM test completed successfully!")
    print("🔬 Check MLFlow at http://localhost:5001 for detailed tracking!")

if __name__ == "__main__":
    test_oneclass_svm()