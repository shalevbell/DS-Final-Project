import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration with sensible defaults."""

    # Database configuration
    DATABASE_URL = os.getenv(
        'DATABASE_URL',
        'postgresql://postgres:postgres@postgres:5432/interview_db'
    )

    # Redis configuration
    REDIS_URL = os.getenv(
        'REDIS_URL',
        'redis://redis:6379/0'
    )

    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = FLASK_ENV == 'development'

    # CORS configuration
    CORS_ORIGINS = '*'

    # Chunk processing configuration
    PROCESSING_MAX_WORKERS = int(os.getenv('PROCESSING_MAX_WORKERS', '3'))
    PROCESSING_QUEUE_SIZE = int(os.getenv('PROCESSING_QUEUE_SIZE', '100'))
    PROCESSING_RETRY_ATTEMPTS = int(os.getenv('PROCESSING_RETRY_ATTEMPTS', '3'))
    PROCESSING_RETRY_DELAY = int(os.getenv('PROCESSING_RETRY_DELAY', '5'))  # seconds

    # PUBSUB configuration
    PUBSUB_CHANNEL = os.getenv('PUBSUB_CHANNEL', 'chunks:ready')
    PUBSUB_RECONNECT_DELAY = int(os.getenv('PUBSUB_RECONNECT_DELAY', '5'))  # seconds

    # Chunk configuration
    CHUNK_DURATION_MS = 30000  # Video chunk duration in milliseconds (30 seconds)
