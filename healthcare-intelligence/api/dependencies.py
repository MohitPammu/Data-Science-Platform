"""
Healthcare Intelligence API Dependencies
Professional dependency injection and service management

Learning Focus: Service architecture and dependency management
"""

import os
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent

def validate_environment():
    """Validate required environment variables and dependencies"""
    required_files = [
        'data/advanced_analytics/ensemble_risk_analysis.csv',
        'streamlit_app.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = get_project_root() / file_path
        if not full_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        raise RuntimeError(f"Missing required files: {missing_files}")
    
    return True

def get_api_metadata():
    """Get API metadata for documentation"""
    return {
        "title": "Healthcare Intelligence API",
        "description": "Real-time provider fraud risk scoring",
        "version": "1.0.0",
        "contact": {
            "name": "Healthcare Intelligence Platform",
            "email": "support@healthcare-intelligence.local"
        }
    }
