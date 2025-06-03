# Deployment Guide - Healthcare Intelligence Platform

## Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- Minimum 4GB RAM, 2 CPU cores
- Network access for container registry

## Development Deployment

### Quick Start
Start services with docker-compose up -d
Verify deployment with docker-compose ps
Check logs with docker-compose logs healthcare_dashboard

### Service Validation
MLflow health check: curl http://localhost:5001/api/2.0/mlflow/experiments/list
Streamlit health check: curl http://localhost:8501/_stcore/health

## Production Deployment

### Environment Configuration
- Configure production environment variables
- Enable resource limits in docker-compose.yml
- Implement SSL/TLS termination
- Configure logging aggregation

### Security Hardening
- Update base images regularly
- Implement secrets management
- Configure network policies
- Enable audit logging

### Monitoring Setup
- Configure health check endpoints
- Implement metrics collection
- Set up alerting for service failures
- Monitor resource utilization

## Troubleshooting

### Common Issues
- Port conflicts: Modify port mappings in docker-compose.yml
- Memory constraints: Increase Docker Desktop memory allocation
- Network issues: Verify healthcare_network configuration
- Permission errors: Check file ownership and Docker daemon access

### Log Analysis
View service logs with docker-compose logs commands
Use docker inspect for container inspection
