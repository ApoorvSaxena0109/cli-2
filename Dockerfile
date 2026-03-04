FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Create data directories
RUN mkdir -p ARS-VG-Analyzer/{input,processed,chromadb,results,graphs}

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/')" || exit 1

# Run the app (no Ollama in container - use --no-ollama)
CMD ["python", "app.py", "--no-ollama", "--port", "7860"]
