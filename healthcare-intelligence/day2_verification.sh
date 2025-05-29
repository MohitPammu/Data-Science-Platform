#!/bin/bash
# DAY 2 VERIFICATION SCRIPT - Run before starting Day 3
# Healthcare Intelligence Platform - Mohit Pammu Portfolio

echo "🔍 DAY 2 VERIFICATION CHECKLIST"
echo "=================================="

# 1. Project Directory Check
echo "📁 Checking project structure..."
cd /Users/mohitpammu/Desktop/data-science-platform/healthcare-intelligence
pwd
echo "✅ Current directory verified"

# 2. Virtual Environment Check
echo "🐍 Checking virtual environment..."
source ../data-science-env/bin/activate
which python
echo "Python version: $(python --version)"
echo "✅ Virtual environment active"

# 3. MLFlow Server Check
echo "🔬 Checking MLFlow server..."
curl -s http://localhost:5001/health && echo "✅ MLFlow server is running" || echo "❌ MLFlow server issue"
curl -s http://localhost:5001/api/2.0/mlflow/experiments/list | head -100

# 4. Data Files Check
echo "📊 Checking data files..."
ls -la data/raw/
echo "Provider data rows: $(tail -n +2 data/raw/medicare_provider_data.csv | wc -l)"
echo "Claims data rows: $(tail -n +2 data/raw/medicare_claims_data.csv | wc -l)"
echo "✅ Data files verified"

# 5. Module Import Check
echo "🔧 Testing Day 2 module imports..."
python -c "
import sys
sys.path.append('.')
print('Testing imports...')

try:
    from src.data_connector.cms_connector import CMSDataConnector
    print('✅ CMSDataConnector imports successfully')
except Exception as e:
    print(f'❌ CMSDataConnector import failed: {e}')

try:
    from src.feature_engineering.fraud_indicators import HealthcareFraudDetector
    print('✅ HealthcareFraudDetector imports successfully')
except Exception as e:
    print(f'❌ HealthcareFraudDetector import failed: {e}')

try:
    from src.geographic_intelligence.geographic_analyzer import GeographicAnalyzer
    print('✅ GeographicAnalyzer imports successfully')
except Exception as e:
    print(f'❌ GeographicAnalyzer import failed: {e}')

try:
    from src.mlflow_tracking.data_tracker import HealthcareDataTracker
    print('✅ HealthcareDataTracker imports successfully')
except Exception as e:
    print(f'❌ HealthcareDataTracker import failed: {e}')

print('Import verification complete!')
"

# 6. Installed Packages Check
echo "📦 Checking required packages..."
pip list | grep -E "(pandas|scikit-learn|mlflow|numpy|matplotlib)"

echo ""
echo "🎯 VERIFICATION COMPLETE"
echo "========================"
echo "If all checks show ✅, you're ready for Day 3 ML development!"
echo "If any checks show ❌, we need to fix those issues first."
