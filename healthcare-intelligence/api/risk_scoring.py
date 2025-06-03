"""
Healthcare Risk Scoring API Integration
Business logic integration with existing platform components

Learning Focus: Service integration, business logic, error handling
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from datetime import datetime
import asyncio
import os
import sys
from pathlib import Path

from .models import ProviderInput, RiskScoreResponse, RiskCategory, InvestigationPriority

logger = logging.getLogger(__name__)

class HealthcareRiskAPI:
    """
    API service for healthcare provider risk scoring
    Learning: Service architecture and business logic integration
    """
    
    def __init__(self):
        """Initialize risk scoring components"""
        self.risk_data = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Load risk scoring data and validate components"""
        try:
            # Load existing risk analysis data
            project_root = Path(__file__).parent.parent
            risk_file = project_root / "data" / "advanced_analytics" / "ensemble_risk_analysis.csv"
            
            if not risk_file.exists():
                raise FileNotFoundError(f"Risk analysis file not found: {risk_file}")
            
            self.risk_data = pd.read_csv(risk_file)
            logger.info(f"Risk scoring initialized with {len(self.risk_data)} provider records")
            
        except Exception as e:
            logger.error(f"Failed to initialize risk scoring components: {str(e)}")
            raise
    
    def validate_components(self) -> bool:
        """Validate that all required components are operational"""
        if self.risk_data is None:
            raise RuntimeError("Risk scoring data not loaded")
        
        if len(self.risk_data) == 0:
            raise RuntimeError("Risk scoring data is empty")
        
        required_columns = ['provider_id', 'composite_risk_score']
        missing_columns = [col for col in required_columns if col not in self.risk_data.columns]
        
        if missing_columns:
            raise RuntimeError(f"Missing required columns: {missing_columns}")
        
        return True
    
    async def calculate_provider_risk(self, provider: ProviderInput) -> RiskScoreResponse:
        """
        Calculate comprehensive risk score for healthcare provider
        
        Learning Focus:
        - Async processing for API responsiveness
        - Business logic integration
        - Error handling and logging
        """
        try:
            # Check if provider exists in our analysis
            existing_provider = self.risk_data[
                self.risk_data['provider_id'] == provider.provider_id
            ]
            
            if not existing_provider.empty:
                # Use existing analysis
                risk_score = float(existing_provider.iloc[0]['composite_risk_score'])
                logger.info(f"Using existing risk score for {provider.provider_id}: {risk_score}")
            else:
                # Calculate new risk score using business logic
                risk_score = self._calculate_new_provider_risk(provider)
                logger.info(f"Calculated new risk score for {provider.provider_id}: {risk_score}")
            
            # Determine risk category and priority
            risk_category, priority = self._categorize_risk(risk_score)
            
            # Calculate business impact
            business_impact = self._calculate_business_impact(provider, risk_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(provider, risk_score)
            
            # Build response
            response = RiskScoreResponse(
                provider_id=provider.provider_id,
                composite_risk_score=risk_score,
                risk_category=risk_category,
                investigation_priority=priority,
                risk_components={
                    'volume_risk': self._calculate_volume_risk(provider),
                    'amount_risk': self._calculate_amount_risk(provider),
                    'diversity_risk': self._calculate_diversity_risk(provider),
                    'specialty_risk': 15.0,  # Default specialty risk
                    'geographic_risk': 10.0   # Default geographic risk
                },
                business_impact=business_impact,
                recommendations=recommendations
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Risk calculation failed for {provider.provider_id}: {str(e)}")
            raise
    
    def _calculate_new_provider_risk(self, provider: ProviderInput) -> float:
        """Calculate risk score for new provider using business logic"""
        # Simplified risk calculation based on provider characteristics
        volume_risk = min((provider.total_claims / 200.0) * 20, 25)  # Max 25 points
        amount_risk = min((provider.avg_claim_amount / 2000.0) * 25, 30)  # Max 30 points
        diversity_risk = min((provider.unique_procedures / 50.0) * 15, 20)  # Max 20 points
        
        # Base risk calculation
        base_risk = volume_risk + amount_risk + diversity_risk
        
        # Add randomization to simulate realistic variation
        risk_score = base_risk + np.random.normal(0, 5)
        
        # Ensure score is within valid range
        return max(0, min(100, risk_score))
    
    def _calculate_volume_risk(self, provider: ProviderInput) -> float:
        """Calculate volume-based risk component"""
        return min((provider.total_claims / 200.0) * 20, 25)
    
    def _calculate_amount_risk(self, provider: ProviderInput) -> float:
        """Calculate amount-based risk component"""
        return min((provider.avg_claim_amount / 2000.0) * 25, 30)
    
    def _calculate_diversity_risk(self, provider: ProviderInput) -> float:
        """Calculate diversity-based risk component"""
        return min((provider.unique_procedures / 50.0) * 15, 20)
    
    def _categorize_risk(self, score: float) -> tuple[RiskCategory, InvestigationPriority]:
        """Categorize risk score into business-meaningful categories"""
        if score >= 80:
            return RiskCategory.CRITICAL, InvestigationPriority.CRITICAL
        elif score >= 60:
            return RiskCategory.HIGH, InvestigationPriority.HIGH
        elif score >= 30:
            return RiskCategory.MEDIUM, InvestigationPriority.MEDIUM
        else:
            return RiskCategory.LOW, InvestigationPriority.LOW
    
    def _calculate_business_impact(self, provider: ProviderInput, risk_score: float) -> Dict[str, Any]:
        """Calculate business impact metrics"""
        # Conservative fraud estimation methodology
        potential_fraud_rate = min(risk_score / 100 * 0.25, 0.20)  # Max 20% fraud rate
        potential_fraud_amount = provider.total_amount * potential_fraud_rate
        investigation_cost = 200  # Standard investigation cost
        
        return {
            'potential_fraud_amount': round(potential_fraud_amount, 2),
            'investigation_cost': investigation_cost,
            'net_benefit': round(potential_fraud_amount * 0.75 - investigation_cost, 2),
            'estimated_fraud_rate': round(potential_fraud_rate * 100, 2),
            'total_exposure': provider.total_amount
        }
    
    def _generate_recommendations(self, provider: ProviderInput, risk_score: float) -> Dict[str, str]:
        """Generate investigation recommendations based on risk components"""
        recommendations = {}
        
        if risk_score >= 80:
            recommendations['priority'] = "Immediate investigation required"
            recommendations['timeline'] = "Within 24 hours"
            recommendations['focus_areas'] = "Claims pattern analysis, billing verification"
        elif risk_score >= 60:
            recommendations['priority'] = "High priority investigation"
            recommendations['timeline'] = "Within 1 week"
            recommendations['focus_areas'] = "Claims review, provider verification"
        elif risk_score >= 30:
            recommendations['priority'] = "Routine monitoring"
            recommendations['timeline'] = "Within 1 month"
            recommendations['focus_areas'] = "Pattern monitoring, periodic review"
        else:
            recommendations['priority'] = "Standard monitoring"
            recommendations['timeline'] = "Quarterly review"
            recommendations['focus_areas'] = "Baseline monitoring"
        
        return recommendations
