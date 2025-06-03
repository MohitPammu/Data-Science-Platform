# Healthcare Intelligence Platform - Optimized Multi-Stage Build
# Production-aware containerization with performance optimization

# Stage 1: Dependencies Installation
FROM python:3.9-slim as dependencies
WORKDIR /app

# Install system dependencies in separate layer for caching
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY docker/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Application Runtime
FROM python:3.9-slim as application
WORKDIR /app

# Copy only necessary files from dependencies stage
COPY --from=dependencies /root/.local /root/.local

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash healthcare
RUN chown -R healthcare:healthcare /app
USER healthcare

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8501

# Health check with proper user context
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
