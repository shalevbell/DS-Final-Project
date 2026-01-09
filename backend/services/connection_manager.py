"""
Connection and service management module.

Handles Redis/Postgres connections, chunk processor initialization, and shutdown logic.
"""

import logging
import sys
import psycopg2
import redis
from typing import Optional, Tuple

from config import Config
from chunk_processor import ChunkProcessor

logger = logging.getLogger(__name__)

# Global Redis client singleton
_redis_client: Optional[redis.Redis] = None

# Global chunk processor instance
_chunk_processor: Optional[ChunkProcessor] = None


def get_redis_client() -> redis.Redis:
    """
    Get Redis client, connecting if needed.

    Returns:
        Redis client instance

    Raises:
        Exception if Redis connection fails
    """
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(Config.REDIS_URL, decode_responses=False)
            _redis_client.ping()
            logger.info('Redis connected')
        except Exception as e:
            logger.error(f'Redis connection failed: {e}')
            _redis_client = None
            raise
    return _redis_client


def check_postgres_health() -> Tuple[bool, str]:
    """
    Test PostgreSQL connection.

    Returns:
        Tuple of (is_connected, message)
    """
    try:
        conn = psycopg2.connect(Config.DATABASE_URL)
        conn.close()
        return True, "Connected"
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        return False, str(e)


def check_redis_health() -> Tuple[bool, str]:
    """
    Test Redis connection.

    Returns:
        Tuple of (is_connected, message)
    """
    try:
        r = redis.from_url(Config.REDIS_URL)
        r.ping()
        return True, "Connected"
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return False, str(e)


def initialize_chunk_processor() -> ChunkProcessor:
    """
    Initialize and start background chunk processor.

    Returns:
        ChunkProcessor instance
    """
    global _chunk_processor
    if _chunk_processor is None:
        _chunk_processor = ChunkProcessor(
            redis_url=Config.REDIS_URL,
            max_workers=Config.PROCESSING_MAX_WORKERS,
            queue_size=Config.PROCESSING_QUEUE_SIZE,
            pubsub_channel=Config.PUBSUB_CHANNEL,
            reconnect_delay=Config.PUBSUB_RECONNECT_DELAY
        )
        _chunk_processor.start()
        logger.info('Chunk processor initialized and started')
    return _chunk_processor


def get_chunk_processor() -> Optional[ChunkProcessor]:
    """
    Get the global chunk processor instance if initialized.

    Returns:
        ChunkProcessor instance or None
    """
    return _chunk_processor


def create_shutdown_handler(processor: ChunkProcessor):
    """
    Create a shutdown handler function for signal handling.

    Args:
        processor: ChunkProcessor instance to shutdown

    Returns:
        Shutdown handler function
    """
    def shutdown_handler(_signum, _frame):
        """Handle graceful shutdown."""
        logger.info('Shutting down gracefully...')
        if processor:
            processor.shutdown()
        sys.exit(0)

    return shutdown_handler
