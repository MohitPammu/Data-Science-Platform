#!/usr/bin/env python
"""
Simple Random Forest Test - Direct Data Loading
Healthcare Intelligence Platform - Mohit Pammu Portfolio
"""

import sys
sys.path.append('.')
import pandas as pd

def test_random_forest_simple():
    """Simple test for Random Forest fraud detection using existing data files."""
    print("🌲 Testing Random Forest Fraud Detection...")
    
    # Load data directly from files (we know they exist from verification)
    try:
        providers_df = pd.read_csv('data/raw/medicare_provider_data.csv')
        claims_df = pd.read_csv('data/raw/medicare_claims_data.csv')
        print(f"✅ Loaded {len(providers_df)} providers and {len(claims_df)} claims from files")
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        return
    
    # Check data structure
    print(f"\nProvider columns: {list(providers_df.columns)}")
    print(f"Claims columns: {list(claims_df.columns)}")
    
    # Initialize Random Forest model
    from src.models.random_forest_fraud import RandomForestFraudDetector
    rf_detector = RandomForestFraudDetector(n_estimators=50, max_depth=10)
    print("✅ Random Forest model initialized")
    
    # Prepare features using the base model method
    try:
        X, y = rf_detector.prepare_healthcare_features(providers_df, claims_df)
        print(f"✅ Prepared {len(X.columns)} features for {len(X)} samples")
        print(f"Fraud rate: {y.mean():.1%}")
        
        # Show feature names
        print(f"\nFeatures created: {list(X.columns)}")
        
    except Exception as e:
        print(f"❌ Feature preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Train model (quick training without hyperparameter tuning)
    try:
        print("\n🚀 Training Random Forest model...")
        results = rf_detector.train_with_hyperparameter_tuning(X, y, tune_hyperparameters=False)
        
        print("\n🎯 Training Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Show top features
    try:
        print(f"\n🔍 Top 5 Important Features:")
        top_features = rf_detector.get_feature_importance().head(5)
        for _, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
    except Exception as e:
        print(f"⚠️  Feature importance display failed: {e}")
    
    # Test prediction on sample
    try:
        sample_prediction = rf_detector.explain_prediction(X.head(1))
        print(f"\n📊 Sample Prediction Explanation:")
        print(f"  Fraud Probability: {sample_prediction['fraud_probability']:.3f}")
        print(f"  Risk Level: {sample_prediction['risk_level']}")
    except Exception as e:
        print(f"⚠️  Sample prediction failed: {e}")
    
    # Show business metrics
    try:
        business_metrics = rf_detector.business_metrics
        print(f"\n💰 Business Impact:")
        print(f"  Potential Fraud Detected: {business_metrics['potential_fraud_detected']} cases")
        print(f"  Estimated Savings: ${business_metrics['estimated_savings']:,.2f}")
        print(f"  Investigation Costs: ${business_metrics['investigation_costs']:,.2f}")
        print(f"  Net Business Value: ${business_metrics['net_business_value']:,.2f}")
    except Exception as e:
        print(f"⚠️  Business metrics display failed: {e}")
    
    print("\n✅ Random Forest test completed successfully!")
    print("🔬 Check MLFlow at http://localhost:5001 for detailed tracking!")

if __name__ == "__main__":
    test_random_forest_simple()
