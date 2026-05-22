FROM python:3.11-slim

# uv: fast Rust-based pip replacement (10-100x faster installs, better caching)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install system dependencies in single layer (better caching)
# Note: no rm -rf of apt lists — cache mounts handle cleanup automatically
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    postgresql-client \
    libglib2.0-0 \
    unzip

# Copy requirements first (better layer caching - changes less frequently than code)
COPY requirements.txt .

# Install Python dependencies with BuildKit cache mount for uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy application code (this layer changes most frequently)
# Note: Whisper model is downloaded at runtime via lazy loading (first use)
COPY . .

# Entrypoint: copy to / so it is NOT overwritten by the ./backend:/app volume (avoids CRLF issues on Windows)
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["/entrypoint.sh"]
# Run with gunicorn using eventlet worker for SocketIO support
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--worker-class", "eventlet", "--workers", "1", "--timeout", "300", "--worker-connections", "1000", "app:app"]
