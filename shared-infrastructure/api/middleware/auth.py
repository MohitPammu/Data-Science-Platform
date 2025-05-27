from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from typing import Optional

security = HTTPBearer(auto_error=False)

class AuthManager:
    def __init__(self):
        self.api_key = os.getenv("API_KEY", "demo-key-12345")
    
    def verify_token(self, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
        if not credentials:
            # For demo purposes, allow requests without auth
            return {"user": "demo", "permissions": ["read"]}
        
        if credentials.credentials != self.api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
        
        return {"user": "authenticated", "permissions": ["read", "write"]}

auth_manager = AuthManager()