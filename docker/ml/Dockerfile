# ML Inference Engine — PyTorch (CPU-optimized)
FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only (lightweight)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir numpy pandas pyarrow fastapi uvicorn[standard] pydantic requests beautifulsoup4 scikit-learn

# Copy full quantflow package
COPY quantflow/ /app/quantflow/
# Override ALL __init__.py files to prevent cascading imports
RUN find /app/quantflow -name "__init__.py" -exec sh -c 'echo "" > {}' \;
COPY docker/ml/server.py /app/server.py

ENV OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
