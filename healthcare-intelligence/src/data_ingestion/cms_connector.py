"""
CMS Data Connector
Handles Medicare/Medicaid dataset downloads and initial processing
"""
import pandas as pd
import requests
import logging
from pathlib import Path
from typing import Dict, List, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CMSDataConnector:
    """Connector for CMS (Centers for Medicare & Medicaid Services) datasets"""
    
    def __init__(self, data_dir: str = "data"):  # FIXED: Remove healthcare-intelligence prefix
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CMS Connector initialized with data directory: {self.data_dir}")
    
    def download_sample_data(self) -> Dict[str, str]:
        """Download sample healthcare datasets for development"""
        logger.info("Setting up sample healthcare datasets...")
        
        # For development, we'll create synthetic datasets based on real CMS structure
        datasets = {}
        
        # Sample Medicare Provider Data
        medicare_data = self._create_sample_medicare_data()
        medicare_path = self.raw_dir / "medicare_provider_data.csv"
        medicare_data.to_csv(medicare_path, index=False)
        datasets['medicare_providers'] = str(medicare_path)
        
        # Sample Claims Data
        claims_data = self._create_sample_claims_data()
        claims_path = self.raw_dir / "medicare_claims_data.csv"
        claims_data.to_csv(claims_path, index=False)
        datasets['claims'] = str(claims_path)
        
        logger.info(f"Created {len(datasets)} sample datasets")
        return datasets
    
    def _create_sample_medicare_data(self) -> pd.DataFrame:
        """Create realistic sample Medicare provider data"""
        import numpy as np
        
        np.random.seed(42)  # For reproducibility
        n_providers = 1000
        
        data = {
            'provider_id': [f'P{str(i).zfill(6)}' for i in range(n_providers)],
            'provider_name': [f'Healthcare Provider {i}' for i in range(n_providers)],
            'state': np.random.choice(['MD', 'VA', 'DC', 'NY', 'CA', 'TX', 'FL'], n_providers),
            'specialty': np.random.choice(['Internal Medicine', 'Cardiology', 'Orthopedics', 'Family Practice'], n_providers),
            'total_claims': np.random.poisson(100, n_providers),
            'total_payment': np.random.lognormal(10, 1, n_providers),
            'avg_claim_amount': np.random.lognormal(6, 0.5, n_providers),
            'claim_count_rank': np.random.randint(1, 1001, n_providers)
        }
        
        return pd.DataFrame(data)
    
    def _create_sample_claims_data(self) -> pd.DataFrame:
        """Create realistic sample claims data"""
        import numpy as np
        from datetime import datetime, timedelta
        
        np.random.seed(42)
        n_claims = 5000
        
        # Generate date range
        start_date = datetime(2023, 1, 1)
        date_range = [start_date + timedelta(days=x) for x in range(365)]
        
        data = {
            'claim_id': [f'C{str(i).zfill(8)}' for i in range(n_claims)],
            'provider_id': [f'P{str(np.random.randint(0, 1000)).zfill(6)}' for _ in range(n_claims)],
            'claim_date': np.random.choice(date_range, n_claims),
            'procedure_code': np.random.choice(['99213', '99214', '99215', '90834', '90837'], n_claims),
            'diagnosis_code': np.random.choice(['M25.511', 'E11.9', 'I10', 'F32.9'], n_claims),
            'claim_amount': np.random.lognormal(5, 0.8, n_claims),
            'paid_amount': lambda x: x * np.random.uniform(0.8, 1.0, len(x)),
            'patient_age': np.random.randint(18, 90, n_claims),
            'service_count': np.random.poisson(2, n_claims)
        }
        
        df = pd.DataFrame(data)
        df['paid_amount'] = df['claim_amount'] * np.random.uniform(0.8, 1.0, n_claims)
        
        return df
