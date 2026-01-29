FROM python:3.11-slim

WORKDIR /app

# Install ffmpeg in its own layer for better caching
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install other system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    postgresql-client \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching - changes less frequently than code)
COPY requirements.txt .

# Install Python dependencies with BuildKit cache mount for pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper model during build (so it's available offline)
# This prevents SSL/connection errors at runtime
# Cache the HuggingFace cache directory so model persists across builds
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from faster_whisper import WhisperModel; import sys; print('Downloading Whisper model...', flush=True); model = WhisperModel('base.en', device='cpu', compute_type='int8'); print('Whisper model downloaded successfully', flush=True); sys.exit(0)" || \
    (echo "Warning: Failed to download Whisper model during build. Will try at runtime." && exit 0)

# Copy application code (this layer changes most frequently)
COPY . .

# Entrypoint: copy to / so it is NOT overwritten by the ./backend:/app volume (avoids CRLF issues on Windows)
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
RUN mkdir -p /data/dataset

# Expose port
EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]
# Run with gunicorn using eventlet worker for SocketIO support
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--worker-class", "eventlet", "--workers", "1", "--timeout", "120", "--worker-connections", "1000", "app:app"]
