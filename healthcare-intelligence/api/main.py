"""
Healthcare Intelligence Platform API
Professional API design demonstrating real-time fraud scoring capability

Learning Focus: RESTful design, input validation, error handling
Business Value: Real-time provider risk assessment for investigation prioritization
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
from typing import Dict, Any
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .models import ProviderInput, RiskScoreResponse, HealthResponse
from .dependencies import validate_environment, get_api_metadata
from .risk_scoring import HealthcareRiskAPI

# Initialize FastAPI application
api_metadata = get_api_metadata()
app = FastAPI(
    title=api_metadata["title"],
    description=api_metadata["description"],
    version=api_metadata["version"],
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit dashboard
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize risk scoring service
risk_api = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global risk_api
    try:
        validate_environment()
        risk_api = HealthcareRiskAPI()
        logger.info("Healthcare Risk API initialized successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    Root endpoint providing API information and navigation
    Learning: API discoverability and professional documentation
    """
    return {
        "message": "Healthcare Intelligence Platform API",
        "description": "Real-time provider fraud risk scoring",
        "endpoints": {
            "health": "/health",
            "documentation": "/docs",
            "risk_scoring": "/api/v1/risk-score"
        },
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for container monitoring and service validation
    Learning: Service monitoring and operational readiness
    """
    try:
        # Verify core components are accessible
        if risk_api:
            risk_api.validate_components()
        
        return HealthResponse(
            status="healthy",
            message="Healthcare Intelligence API operational",
            version="1.0.0",
            components={
                "risk_scoring": "operational" if risk_api else "initializing",
                "data_pipeline": "ready",
                "model_registry": "accessible"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/api/v1/risk-score", response_model=RiskScoreResponse)
async def score_provider_risk(provider: ProviderInput):
    """
    Score individual provider fraud risk in real-time
    
    Learning Focus:
    - Input validation and error handling
    - Business logic integration
    - Professional API design patterns
    
    Business Value:
    - Real-time fraud risk assessment
    - Investigation prioritization
    - Regulatory compliance support
    """
    try:
        logger.info(f"Processing risk score request for provider {provider.provider_id}")
        
        if not risk_api:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Risk scoring service not initialized"
            )
        
        # Calculate risk score using integrated platform
        risk_result = await risk_api.calculate_provider_risk(provider)
        
        logger.info(f"Risk score calculated for {provider.provider_id}: {risk_result.composite_risk_score}")
        return risk_result
        
    except ValueError as ve:
        logger.warning(f"Validation error for provider {provider.provider_id}: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid provider data: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Risk scoring failed for provider {provider.provider_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk scoring service temporarily unavailable"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for professional error management
    Learning: Comprehensive error handling and logging
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
