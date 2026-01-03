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
