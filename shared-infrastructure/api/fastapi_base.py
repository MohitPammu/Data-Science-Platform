from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Dict, Any
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAPI:
    def __init__(self, title: str, description: str, version: str = "1.0.0"):
        self.app = FastAPI(
            title=title,
            description=description,
            version=version
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add middleware
        self.app.middleware("http")(self.log_requests)
        
        # Add basic routes
        self.setup_basic_routes()
    
    async def log_requests(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.4f}s"
        )
        return response
    
    def setup_basic_routes(self):
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "Healthcare Intelligence API"
            }
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Healthcare Operations Intelligence Platform",
                "version": "1.0.0",
                "endpoints": ["/health", "/docs", "/fraud-detection", "/provider-analytics"]
            }

# Base response models
class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    risk_level: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str