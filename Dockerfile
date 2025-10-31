FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# System deps (optional; keep minimal)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app ./app
COPY allowed_categories.json ./allowed_categories.json

EXPOSE 8080

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

