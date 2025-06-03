# Healthcare Intelligence API

## Overview
Professional FastAPI implementation demonstrating real-time fraud risk scoring capability for healthcare providers.

## Learning Objectives
- RESTful API design and implementation
- Professional input validation and error handling
- Business logic integration with existing analytics platform
- Real-time service architecture and performance

## API Endpoints

### GET /
Root endpoint providing API information and navigation

### GET /health
Health check endpoint for monitoring and container orchestration

### POST /api/v1/risk-score
Real-time provider fraud risk scoring

**Request Format:**
```json
{
  "provider_id": "PROV_12345",
  "specialty": "Cardiology", 
  "state": "CA",
  "total_claims": 250,
  "total_amount": 450000.0,
  "avg_claim_amount": 1800.0,
  "unique_procedures": 35,
  "unique_diagnoses": 40
}


