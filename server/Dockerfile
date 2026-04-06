# ============================================================================
# Dockerfile — Code Review OpenEnv Environment
#
# Build:   docker build -t code-review-env -f server/Dockerfile .
# Run:     docker run -p 7860:7860 code-review-env
# ============================================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="code-review-env"
LABEL description="OpenEnv Code Review Environment — evaluates AI code review agents"

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker cache efficiency)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the default port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
