# Multi-stage isn't worth it here: the slow part is the embedding model
# download, which we trigger at build time so runtime doesn't need network.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System libs needed by faiss-cpu / sentence-transformers (torch).
# `build-essential` only matters if a wheel is missing; it's small enough to keep.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so the model-download layer reuses the
# cached dependency layer when only the model code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the embedding model into the default HF cache location.
# Saves ~30s+ on the first request and means the container can run
# fully offline at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the app source last.
COPY . .

# HuggingFace expects 7860 by default for its Spaces / API.
EXPOSE 7860

# Single worker: the FAISS index and embedding model live in process
# memory, and concurrent requests would just serialize on the GIL anyway.
# If you ever need more throughput, move to a separate inference server.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
