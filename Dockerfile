# Use a lightweight Python 3.12 image.
FROM python:3.12-slim

# Prevent Python from generating unnecessary cache files.
ENV PYTHONDONTWRITEBYTECODE=1

# Display Python logs immediately.
ENV PYTHONUNBUFFERED=1

# Configure Hugging Face model cache.
ENV HF_HOME=/app/.cache/huggingface

# Set the working directory.
WORKDIR /app

# Install the system library required by some ML packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions first for better build caching.
COPY requirements.txt .

# Upgrade pip.
RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch for ML and embedding operations.
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install the remaining project dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code.
COPY app/ ./app/
COPY ml/ ./ml/
COPY ml_artifacts/ ./ml_artifacts/

# Include Alembic configuration and database migration scripts.
COPY alembic.ini ./alembic.ini
COPY alembic/ ./alembic/

# Create directories for persistent application data.
RUN mkdir -p /app/data /app/vector_db /app/.cache/huggingface

# Create a non-root application user.
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

# Run the application without root privileges.
USER appuser

# Expose the FastAPI port.
EXPOSE 8000

# Apply database migrations before starting the FastAPI server.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]