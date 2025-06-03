"""
Healthcare Intelligence API Data Models
Professional Pydantic models for input validation and response formatting

Learning Focus: Data validation, API design patterns, type safety
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, Any
from datetime import datetime
from enum import Enum

class RiskCategory(str, Enum):
    """Risk classification categories"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class InvestigationPriority(str, Enum):
    """Investigation priority levels"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class ProviderInput(BaseModel):
    """
    Input model for provider risk scoring
    Learning: Professional input validation with business rules
    """
    provider_id: str = Field(..., description="Unique provider identifier", max_length=50)
    specialty: str = Field(..., description="Healthcare specialty", max_length=100)
    state: str = Field(..., description="Provider state (2-letter code)", min_length=2, max_length=2)
    total_claims: int = Field(..., ge=0, description="Total number of claims submitted")
    total_amount: float = Field(..., ge=0, description="Total claim amount in USD")
    avg_claim_amount: float = Field(..., ge=0, description="Average claim amount in USD")
    unique_procedures: int = Field(..., ge=0, description="Number of unique procedure codes")
    unique_diagnoses: int = Field(..., ge=0, description="Number of unique diagnosis codes")
    
    @validator('state')
    def validate_state_code(cls, v):
        """Validate US state codes"""
        valid_states = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
        }
        if v.upper() not in valid_states:
            raise ValueError(f'Invalid state code: {v}')
        return v.upper()
    
    @validator('avg_claim_amount')
    def validate_avg_claim_consistency(cls, v, values):
        """Ensure average claim amount is consistent with totals"""
        if 'total_claims' in values and 'total_amount' in values:
            if values['total_claims'] > 0:
                calculated_avg = values['total_amount'] / values['total_claims']
                if abs(v - calculated_avg) > 1.0:  # Allow $1 rounding difference
                    raise ValueError('Average claim amount inconsistent with total claims and amount')
        return v

class RiskScoreResponse(BaseModel):
    """
    Response model for provider risk scoring
    Learning: Professional response formatting with business context
    """
    provider_id: str
    composite_risk_score: float = Field(..., ge=0, le=100, description="Overall risk score (0-100)")
    risk_category: RiskCategory
    investigation_priority: InvestigationPriority
    risk_components: Dict[str, float] = Field(..., description="Individual risk component scores")
    business_impact: Dict[str, Any] = Field(..., description="Business impact analysis")
    recommendations: Dict[str, str] = Field(..., description="Investigation recommendations")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        """Pydantic configuration for professional JSON serialization"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HealthResponse(BaseModel):
    """
    Health check response model
    Learning: Service monitoring and operational validation
    """
    status: str
    message: str
    version: str
    components: Optional[Dict[str, str]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
