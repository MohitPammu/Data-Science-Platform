"""
Healthcare Provider Risk Scoring System
Comprehensive provider risk assessment using clustering and fraud indicators
Advanced analytics module for fraud detection platform
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Import base model class and clustering analyzer
import sys
sys.path.append('.')
from src.models.base_healthcare_model import BaseHealthcareModel

class ProviderRiskScorer(BaseHealthcareModel):
    """
    Comprehensive provider risk scoring system
    Integrates clustering analysis with fraud indicators for 0-100 risk scores
    """
    
    def __init__(self):
        super().__init__("ProviderRiskScoring", "Advanced_Analytics")
        self.risk_weights = {
            'cluster_risk': 0.30,      # Cluster-based risk assessment
            'volume_risk': 0.20,       # Billing volume patterns
            'amount_risk': 0.25,       # Claim amount patterns
            'specialty_risk': 0.15,    # Specialty-specific risks
            'diversity_risk': 0.10     # Procedure/diagnosis diversity
        }
        self.scaler = MinMaxScaler()
        self.risk_thresholds = {
            'low': 30,
            'medium': 60,
            'high': 80
        }
        
    def calculate_cluster_risk_scores(self, clustered_data: pd.DataFrame, cluster_analysis: dict) -> pd.DataFrame:
        """
        Calculate risk scores based on cluster characteristics
        
        Args:
            clustered_data: DataFrame with cluster assignments
            cluster_analysis: Analysis results from clustering
            
        Returns:
            DataFrame with cluster risk scores
        """
        self.logger.info("Calculating cluster-based risk scores")
        
        try:
            # Create cluster risk mapping based on analysis
            cluster_risk_mapping = {}
            
            # Analyze each cluster's risk characteristics
            for cluster_id in sorted(clustered_data['cluster'].unique()):
                cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
                
                # Calculate risk indicators
                avg_claim_amount = cluster_data['avg_claim_amount'].mean()
                total_volume = cluster_data['total_claims'].mean()
                amount_std = cluster_data['claim_amount_std'].mean()
                
                # Percentile-based risk assessment
                amount_percentile = (clustered_data['avg_claim_amount'] <= avg_claim_amount).mean() * 100
                volume_percentile = (clustered_data['total_claims'] <= total_volume).mean() * 100
                
                # Calculate composite cluster risk (0-100 scale)
                cluster_risk = (
                    amount_percentile * 0.6 +  # Higher amounts = higher risk
                    volume_percentile * 0.3 +  # Higher volume = higher risk
                    min(amount_std / avg_claim_amount * 100, 50) * 0.1  # Higher variability = higher risk
                )
                
                cluster_risk_mapping[cluster_id] = min(cluster_risk, 100)
            
            # Assign cluster risk scores to providers
            clustered_data['cluster_risk_score'] = clustered_data['cluster'].map(cluster_risk_mapping)
            
            self.logger.info(f"Cluster risk scores calculated: {cluster_risk_mapping}")
            
            return clustered_data
            
        except Exception as e:
            self.logger.error(f"Error calculating cluster risk scores: {str(e)}")
            raise
    
    def calculate_volume_risk_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk scores based on billing volume patterns
        
        Args:
            data: DataFrame with provider data
            
        Returns:
            DataFrame with volume risk scores
        """
        self.logger.info("Calculating volume-based risk scores")
        
        try:
            # Enhanced percentile-based scaling for realistic distribution
            volume_percentiles = data['total_claims'].rank(pct=True)
        
            # Apply realistic healthcare fraud distribution
            volume_risk_scores = np.where(
                volume_percentiles >= 0.95,  # Top 5% get 85-100 (critical)
                85 + (volume_percentiles - 0.95) / 0.05 * 15,
                np.where(
                    volume_percentiles >= 0.80,  # Next 15% get 60-85 (high)
                    60 + (volume_percentiles - 0.80) / 0.15 * 25,
                    np.where(
                        volume_percentiles >= 0.20,  # Middle 60% get 10-60 (medium)
                        10 + (volume_percentiles - 0.20) / 0.60 * 50,
                        np.maximum(0, (volume_percentiles - 0.001) / 0.199 * 10)
                    )
                )
            )
        
            # High volume bonus for extreme outliers
            high_volume_threshold = data['total_claims'].quantile(0.95)
            volume_risk_scores = np.where(
                data['total_claims'] > high_volume_threshold,
                np.minimum(volume_risk_scores + 10, 100),
                volume_risk_scores
            )

            # Ensure we have at least one 0 score for the lowest provider
            if len(volume_risk_scores) > 0:
                min_idx = data['total_claims'].idxmin()
                volume_risk_scores[data.index.get_loc(min_idx)] = 0.0
        
            data['volume_risk_score'] = volume_risk_scores.round(1)
            self.logger.info("Volume risk scores calculated")
            return data
        
        except Exception as e:
            self.logger.error(f"Error calculating volume risk scores: {str(e)}")
            raise
    
    def calculate_amount_risk_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk scores based on claim amount patterns
        
        Args:
            data: DataFrame with provider data
            
        Returns:
            DataFrame with amount risk scores
        """
        self.logger.info("Calculating amount-based risk scores")
        
        try:
            # Enhanced percentile-based scaling
            amount_percentiles = data['avg_claim_amount'].rank(pct=True)
        
            # Apply realistic healthcare fraud distribution
            amount_risk_scores = np.where(
                amount_percentiles >= 0.95,  # Top 5% get 85-100 (critical)
                85 + (amount_percentiles - 0.95) / 0.05 * 15,
                np.where(
                    amount_percentiles >= 0.80,  # Next 15% get 60-85 (high)
                    60 + (amount_percentiles - 0.80) / 0.15 * 25,
                    np.where(
                        amount_percentiles >= 0.20,  # Middle 60% get 10-60 (medium)
                        10 + (amount_percentiles - 0.20) / 0.60 * 50,
                        np.maximum(0, (amount_percentiles - 0.001) / 0.199 * 10)
                    )
                )
            )
        
            # High amount bonus for extreme cases
            high_amount_threshold = data['avg_claim_amount'].quantile(0.90)
            amount_risk_scores = np.where(
                data['avg_claim_amount'] > high_amount_threshold,
                np.minimum(amount_risk_scores + 15, 100),
                amount_risk_scores
            )

            # Ensure we have at least one 0 score for the lowest provider
            if len(amount_risk_scores) > 0:
                min_idx = data['avg_claim_amount'].idxmin()
                amount_risk_scores[data.index.get_loc(min_idx)] = 0.0
        
            data['amount_risk_score'] = amount_risk_scores.round(1)
            self.logger.info("Amount risk scores calculated")
            return data
        
        except Exception as e:
            self.logger.error(f"Error calculating amount risk scores: {str(e)}")
            raise
    
    def calculate_specialty_risk_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk scores based on specialty patterns
        
        Args:
            data: DataFrame with provider data
            
        Returns:
            DataFrame with specialty risk scores
        """
        self.logger.info("Calculating specialty-based risk scores")
        
        try:
            # Define specialty risk levels based on healthcare fraud patterns
            specialty_risk_levels = {
                'Internal Medicine': 70,
                'Cardiology': 65,
                'Orthopedic Surgery': 60,
                'Neurology': 55,
                'Emergency Medicine': 50,
                'Family Medicine': 45,
                'Pediatrics': 40,
                'Psychiatry': 35,
                'Dermatology': 30
            }
            
            # Assign specialty risk scores
            data['specialty_base_risk'] = data['specialty'].map(specialty_risk_levels).fillna(50)
            
            # Adjust based on specialty concentration in geographic area
            specialty_concentration = data.groupby(['state', 'specialty']).size() / data.groupby('state').size()
            specialty_concentration_map = specialty_concentration.to_dict()
            
            # Create adjustment factor (higher concentration = lower individual risk)
            data['specialty_concentration'] = data.apply(
                lambda row: specialty_concentration_map.get((row['state'], row['specialty']), 0.1), 
                axis=1
            )
            
            # Final specialty risk score
            data['specialty_risk_score'] = data['specialty_base_risk'] * (1 - data['specialty_concentration'] * 0.3)
            
            self.logger.info("Specialty risk scores calculated")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating specialty risk scores: {str(e)}")
            raise
    
    def calculate_diversity_risk_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk scores based on procedure/diagnosis diversity
        
        Args:
            data: DataFrame with provider data
            
        Returns:
            DataFrame with diversity risk scores
        """
        self.logger.info("Calculating diversity-based risk scores")
        
        try:
            # Procedure diversity risk
            data['procedure_concentration'] = data['unique_procedures'] / data['total_claims']
            data['proc_concentration_percentile'] = data['procedure_concentration'].rank(pct=True) * 100
            
            # Diagnosis diversity risk  
            data['diagnosis_concentration'] = data['unique_diagnoses'] / data['total_claims']
            data['diag_concentration_percentile'] = data['diagnosis_concentration'].rank(pct=True) * 100
            
            # Higher concentration (less diversity) = higher risk for fraud
            data['diversity_risk_score'] = 100 - (
                data['proc_concentration_percentile'] * 0.6 +
                data['diag_concentration_percentile'] * 0.4
            )
            
            self.logger.info("Diversity risk scores calculated")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating diversity risk scores: {str(e)}")
            raise
    
    def calculate_composite_risk_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate final composite risk score using weighted components
        
        Args:
            data: DataFrame with individual risk components
            
        Returns:
            DataFrame with composite risk scores
        """
        self.logger.info("Calculating composite risk scores")
        
        try:
            # Ensure all component scores exist
            required_columns = [
                'cluster_risk_score', 'volume_risk_score', 'amount_risk_score',
                'specialty_risk_score', 'diversity_risk_score'
            ]
        
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing risk score columns: {missing_columns}")
        
            # Calculate weighted composite score
            composite_raw = (
                data['cluster_risk_score'] * self.risk_weights['cluster_risk'] +
                data['volume_risk_score'] * self.risk_weights['volume_risk'] +
                data['amount_risk_score'] * self.risk_weights['amount_risk'] +
                data['specialty_risk_score'] * self.risk_weights['specialty_risk'] +
                data['diversity_risk_score'] * self.risk_weights['diversity_risk']
            )
        
            # Final calibration for realistic business distribution
            composite_percentiles = composite_raw.rank(pct=True)
        
            # Target: ~11% high risk (≥80), ~5% critical (≥90)
            calibrated_scores = np.where(
                composite_percentiles >= 0.95,  # Top 5% get 90-100 (critical)
                90 + (composite_percentiles - 0.95) / 0.05 * 10,
                np.where(
                    composite_percentiles >= 0.89,  # Next 6% get 80-90 (high)
                    80 + (composite_percentiles - 0.89) / 0.06 * 10,
                    np.where(
                        composite_percentiles >= 0.20,  # Middle 69% get 10-80 (medium)
                        10 + (composite_percentiles - 0.20) / 0.69 * 70,
                        np.maximum(0, (composite_percentiles - 0.001) / 0.199 * 10)
                    )
                )
            )

            # Ensure we have at least one 0 score for the absolute lowest risk provider
            if len(calibrated_scores) > 0:
                min_idx = composite_raw.idxmin()
                calibrated_scores[data.index.get_loc(min_idx)] = 0.0
        
            # Set final scores and ensure bounds
            data['composite_risk_score'] = np.clip(calibrated_scores, 0, 100).round(1)
        
            # Update risk categories to match actual scores
            data['risk_category'] = pd.cut(
                data['composite_risk_score'],
                bins=[0, 30, 60, 80, 100],
                labels=['Low', 'Medium', 'High', 'Critical'],
                include_lowest=True
            )
        
            self.logger.info("Composite risk scores calculated")
            return data
        
        except Exception as e:
            self.logger.error(f"Error calculating composite risk scores: {str(e)}")
            raise