"""
Healthcare Intelligence API Testing
Professional API validation and testing framework

Learning Focus: API testing, validation, and quality assurance
"""

import requests
import json
import time
from datetime import datetime

def test_api_endpoints():
    """Test all API endpoints for functionality"""
    base_url = "http://localhost:8000"
    
    print("Healthcare Intelligence API Testing")
    print("==================================")
    
    # Test 1: Root endpoint
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("PASS: Root endpoint accessible")
        else:
            print(f"FAIL: Root endpoint returned {response.status_code}")
    except Exception as e:
        print(f"FAIL: Root endpoint error - {str(e)}")
    
    # Test 2: Health check
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("PASS: Health check endpoint operational")
        else:
            print(f"FAIL: Health check returned {response.status_code}")
    except Exception as e:
        print(f"FAIL: Health check error - {str(e)}")
    
    # Test 3: Risk scoring endpoint
    test_payload = {
        "provider_id": "API_TEST_001",
        "specialty": "Internal Medicine",
        "state": "MD",
        "total_claims": 150,
        "total_amount": 225000.0,
        "avg_claim_amount": 1500.0,
        "unique_procedures": 25,
        "unique_diagnoses": 30
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/v1/risk-score",
            json=test_payload,
            timeout=30
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            risk_score = result.get('composite_risk_score', 0)
            print(f"PASS: Risk scoring endpoint - Score: {risk_score:.2f} in {response_time:.3f}s")
        else:
            print(f"FAIL: Risk scoring returned {response.status_code}")
    except Exception as e:
        print(f"FAIL: Risk scoring error - {str(e)}")
    
    print("==================================")
    print("API testing complete")

if __name__ == "__main__":
    test_api_endpoints()
