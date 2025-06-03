# API Module - Development
## FastAPI Implementation

This module contains the production API code for real-time fraud risk scoring.

### Development Files:
- `main.py` - FastAPI application and endpoints
- `models.py` - Pydantic data models and validation
- `risk_scoring.py` - Business logic integration
- `dependencies.py` - Service dependencies
- `test_api.py` - API testing framework

### Professional Documentation:
See `/docs/api/API_DOCUMENTATION.md` for client-facing API documentation.

### Local Development:
```bash
uvicorn api.main:app --reload --port 8000
