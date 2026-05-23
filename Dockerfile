FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source
COPY . .

# Default: run FastAPI (overridden per-process in fly.toml)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
