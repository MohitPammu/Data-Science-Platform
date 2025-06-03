#!/usr/bin/env python
"""
Overfitting Analysis for XGBoost Healthcare Fraud Detection
Healthcare Intelligence Platform - Mohit Pammu Portfolio

Comprehensive analysis to identify potential overfitting in our models.
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_overfitting():
    """Comprehensive overfitting analysis for our fraud detection models."""
    print("🔍 OVERFITTING ANALYSIS FOR HEALTHCARE FRAUD DETECTION")
    print("=" * 60)
    
    # Load data
    providers_df = pd.read_csv('data/raw/medicare_provider_data.csv')
    claims_df = pd.read_csv('data/raw/medicare_claims_data.csv')
    print(f"✅ Loaded {len(providers_df)} providers and {len(claims_df)} claims")
    
    # Prepare features using our base model
    from src.models.base_healthcare_model import BaseHealthcareModel
    base_model = BaseHealthcareModel("Analysis", "Overfitting_Analysis")
    X, y = base_model.prepare_healthcare_features(providers_df, claims_df)
    
    print(f"✅ Prepared {len(X.columns)} features for {len(X)} samples")
    print(f"Fraud rate: {y.mean():.1%}")
    
    # 1. DATA LEAKAGE ANALYSIS
    print("\n🕵️ 1. DATA LEAKAGE ANALYSIS")
    print("-" * 30)
    
    # Check if target variable can be perfectly predicted from features
    print("Feature correlations with target (fraud labels):")
    correlations = []
    for col in X.columns:
        corr = X[col].corr(y)
        if abs(corr) > 0.8:  # High correlation threshold
            correlations.append((col, corr))
            
    if correlations:
        print("⚠️  POTENTIAL DATA LEAKAGE DETECTED:")
        for col, corr in sorted(correlations, key=lambda x: abs(x[1]), reverse=True):
            print(f"  {col}: {corr:.3f}")
    else:
        print("✅ No obvious data leakage detected")
    
    # 2. TRAIN-TEST SPLIT ANALYSIS
    print("\n📊 2. TRAIN-TEST SPLIT ANALYSIS")
    print("-" * 30)
    
    # Split data properly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples ({y_train.mean():.1%} fraud)")
    print(f"Test set: {len(X_test)} samples ({y_test.mean():.1%} fraud)")
    
    # Test both models on hold-out data
    from src.models.random_forest_fraud import RandomForestFraudDetector
    from src.models.xgboost_fraud import XGBoostFraudDetector
    
    # 3. RANDOM FOREST VALIDATION
    print("\n🌲 3. RANDOM FOREST HOLD-OUT VALIDATION")
    print("-" * 40)
    
    rf_model = RandomForestFraudDetector(n_estimators=50, max_depth=10)
    rf_model.model.fit(X_train, y_train)
    rf_model.is_trained = True
    
    # Predictions on training and test sets
    rf_train_pred = rf_model.model.predict(X_train)
    rf_test_pred = rf_model.model.predict(X_test)
    
    rf_train_f1 = f1_score(y_train, rf_train_pred)
    rf_test_f1 = f1_score(y_test, rf_test_pred)
    
    print(f"Random Forest Training F1: {rf_train_f1:.3f}")
    print(f"Random Forest Test F1: {rf_test_f1:.3f}")
    print(f"Performance Gap: {rf_train_f1 - rf_test_f1:.3f}")
    
    if rf_train_f1 - rf_test_f1 > 0.1:
        print("⚠️  Potential overfitting detected (gap > 0.1)")
    else:
        print("✅ No significant overfitting detected")
    
    # 4. XGBOOST VALIDATION
    print("\n🚀 4. XGBOOST HOLD-OUT VALIDATION")
    print("-" * 35)
    
    xgb_model = XGBoostFraudDetector(n_estimators=100, max_depth=6)
    xgb_model.model.fit(X_train, y_train)
    xgb_model.is_trained = True
    
    # Predictions on training and test sets
    xgb_train_pred = xgb_model.model.predict(X_train)
    xgb_test_pred = xgb_model.model.predict(X_test)
    
    xgb_train_f1 = f1_score(y_train, xgb_train_pred)
    xgb_test_f1 = f1_score(y_test, xgb_test_pred)
    
    print(f"XGBoost Training F1: {xgb_train_f1:.3f}")
    print(f"XGBoost Test F1: {xgb_test_f1:.3f}")
    print(f"Performance Gap: {xgb_train_f1 - xgb_test_f1:.3f}")
    
    if xgb_train_f1 - xgb_test_f1 > 0.1:
        print("⚠️  Potential overfitting detected (gap > 0.1)")
    else:
        print("✅ No significant overfitting detected")
    
    # 5. CROSS-VALIDATION DEEPER ANALYSIS
    print("\n📈 5. CROSS-VALIDATION STABILITY ANALYSIS")
    print("-" * 45)
    
    # Multiple CV runs to check stability
    cv_scores_rf = []
    cv_scores_xgb = []
    
    for i in range(5):
        # Random Forest CV
        rf_cv = cross_val_score(rf_model.model, X, y, cv=5, scoring='f1')
        cv_scores_rf.extend(rf_cv)
        
        # XGBoost CV
        xgb_cv = cross_val_score(xgb_model.model, X, y, cv=5, scoring='f1')
        cv_scores_xgb.extend(xgb_cv)
    
    rf_mean = np.mean(cv_scores_rf)
    rf_std = np.std(cv_scores_rf)
    xgb_mean = np.mean(cv_scores_xgb)
    xgb_std = np.std(cv_scores_xgb)
    
    print(f"Random Forest CV F1: {rf_mean:.3f} ± {rf_std:.3f}")
    print(f"XGBoost CV F1: {xgb_mean:.3f} ± {xgb_std:.3f}")
    
    if rf_std > 0.1 or xgb_std > 0.1:
        print("⚠️  High variance detected - potential instability")
    else:
        print("✅ Models show stable performance across CV folds")
    
    # 6. FEATURE ANALYSIS - FIXED
    print("\n🔍 6. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 35)
    
    # Check if fraud labels were created correctly
    print("Fraud label distribution analysis:")
    fraud_summary = y.value_counts()
    print(f"Legitimate (0): {fraud_summary.get(0, 0)} ({fraud_summary.get(0, 0)/len(y):.1%})")
    print(f"Fraud (1): {fraud_summary.get(1, 0)} ({fraud_summary.get(1, 0)/len(y):.1%})")
    
    # Check fraud label creation logic
    print("\nFraud label creation analysis:")
    payment_threshold = providers_df['total_payment'].quantile(0.8)
    claim_threshold = providers_df['total_claims'].quantile(0.85)
    
    high_payment = (providers_df['total_payment'] >= payment_threshold).sum()
    high_claims = (providers_df['total_claims'] >= claim_threshold).sum()
    
    print(f"High payment providers (>80th percentile): {high_payment}")
    print(f"High claims providers (>85th percentile): {high_claims}")
    print(f"Overlap: This creates the fraud labels")
    
    # 7. RECOMMENDATIONS
    print("\n🎯 7. RECOMMENDATIONS")
    print("-" * 25)
    
    if xgb_mean > 0.95:
        print("⚠️  VERY HIGH F1 SCORES DETECTED")
        print("Potential causes:")
        print("  1. Labels based on features used for prediction (data leakage)")
        print("  2. Fraud pattern is too obvious in this dataset")
        print("  3. Dataset is too clean/synthetic")
        print("\nRecommendations:")
        print("  1. Use external fraud labels if available")
        print("  2. Make fraud detection more challenging")
        print("  3. Add noise/complexity to labels")
        print("  4. Focus on business value metrics over technical scores")
    else:
        print("✅ Performance levels appear reasonable")
    
    print("\n📊 Creating performance comparison visualization...")
    
    # FIXED VISUALIZATION - Handle NaN values
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Train vs Test Performance
    plt.subplot(2, 2, 1)
    models = ['Random Forest', 'XGBoost']
    train_scores = [rf_train_f1, xgb_train_f1]
    test_scores = [rf_test_f1, xgb_test_f1]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, train_scores, width, label='Training F1', alpha=0.8)
    plt.bar(x + width/2, test_scores, width, label='Test F1', alpha=0.8)
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.title('Training vs Test Performance')
    plt.xticks(x, models)
    plt.legend()
    plt.ylim(0, 1.1)
    
    # Subplot 2: CV Score Distribution - FIXED
    plt.subplot(2, 2, 2)
    plt.boxplot([cv_scores_rf, cv_scores_xgb], tick_labels=['Random Forest', 'XGBoost'])
    plt.ylabel('F1 Score')
    plt.title('Cross-Validation Score Distribution')
    
    # Subplot 3: Feature Correlation with Target - FIXED
    plt.subplot(2, 2, 3)
    feature_corrs = []
    for col in X.columns:
        corr = X[col].corr(y)
        if not np.isnan(corr):  # Only include non-NaN correlations
            feature_corrs.append(corr)
    
    if feature_corrs:  # Only plot if we have valid correlations
        plt.hist(feature_corrs, bins=20, alpha=0.7)
        plt.xlabel('Correlation with Fraud Labels')
        plt.ylabel('Number of Features')
        plt.title('Feature-Target Correlation Distribution')
    else:
        plt.text(0.5, 0.5, 'No valid correlations found', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.title('Feature-Target Correlation Distribution')
    
    # Subplot 4: Fraud Label Distribution
    plt.subplot(2, 2, 4)
    plt.pie([fraud_summary.get(0, 0), fraud_summary.get(1, 0)], 
            labels=['Legitimate', 'Fraud'], autopct='%1.1f%%')
    plt.title('Fraud Label Distribution')
    
    plt.tight_layout()
    plt.savefig('models/artifacts/overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Overfitting analysis complete!")
    print("📊 Analysis saved to: models/artifacts/overfitting_analysis.png")

if __name__ == "__main__":
    analyze_overfitting()
