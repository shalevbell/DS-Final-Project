"""
Redis helper utilities for chunk processing.

Provides key formatting, connection management, and common Redis operations.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def format_session_key(session_id: str, suffix: str) -> str:
    """
    Format a session-level Redis key.

    Args:
        session_id: Session identifier
        suffix: Key suffix (e.g., 'info', 'chunks')

    Returns:
        Formatted key: 'session:{sessionId}:{suffix}'
    """
    return f'session:{session_id}:{suffix}'


def format_chunk_key(session_id: str, chunk_index: int, suffix: str) -> str:
    """
    Format a chunk-level Redis key.

    Args:
        session_id: Session identifier
        chunk_index: Chunk index within session
        suffix: Key suffix (e.g., 'video', 'audio', 'meta', 'status', 'results:whisper')

    Returns:
        Formatted key: 'session:{sessionId}:chunk:{chunkIndex}:{suffix}'
    """
    return f'session:{session_id}:chunk:{chunk_index}:{suffix}'


def get_chunk_video_key(session_id: str, chunk_index: int) -> str:
    """Get Redis key for chunk video data."""
    return format_chunk_key(session_id, chunk_index, 'video')


def get_chunk_audio_key(session_id: str, chunk_index: int) -> str:
    """Get Redis key for chunk audio data."""
    return format_chunk_key(session_id, chunk_index, 'audio')


def get_chunk_meta_key(session_id: str, chunk_index: int) -> str:
    """Get Redis key for chunk metadata."""
    return format_chunk_key(session_id, chunk_index, 'meta')


def get_chunk_status_key(session_id: str, chunk_index: int) -> str:
    """Get Redis key for chunk processing status."""
    return format_chunk_key(session_id, chunk_index, 'status')


def get_chunk_results_key(session_id: str, chunk_index: int, model: str) -> str:
    """
    Get Redis key for chunk analysis results.

    Args:
        session_id: Session identifier
        chunk_index: Chunk index within session
        model: Model name ('whisper', 'mediapipe', 'vocaltone')

    Returns:
        Formatted key: 'session:{sessionId}:chunk:{chunkIndex}:results:{model}'
    """
    return format_chunk_key(session_id, chunk_index, f'results:{model}')


def get_chunk_error_key(session_id: str, chunk_index: int) -> str:
    """Get Redis key for chunk error information."""
    return format_chunk_key(session_id, chunk_index, 'error')


def get_session_chunks_key(session_id: str) -> str:
    """Get Redis key for session chunk list."""
    return format_session_key(session_id, 'chunks')


def get_session_info_key(session_id: str) -> str:
    """Get Redis key for session metadata."""
    return format_session_key(session_id, 'info')


class RedisConnectionManager:
    """
    Manages Redis connections with error handling and reconnection logic.
    """

    def __init__(self, redis_url: str):
        """
        Initialize Redis connection manager.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.client: Optional[any] = None

    def connect(self, decode_responses: bool = False):
        """
        Create Redis client connection.

        Args:
            decode_responses: Whether to decode responses to strings (default: False for binary data)

        Returns:
            Redis client instance

        Raises:
            Exception if connection fails
        """
        import redis

        try:
            self.client = redis.from_url(self.redis_url, decode_responses=decode_responses)
            self.client.ping()
            logger.info(f'Redis connected: {self.redis_url}')
            return self.client
        except Exception as e:
            logger.error(f'Redis connection failed: {e}')
            raise

    def get_client(self):
        """
        Get Redis client, connecting if necessary.

        Returns:
            Redis client instance
        """
        if self.client is None:
            self.connect()
        return self.client

    def close(self):
        """Close Redis connection."""
        if self.client:
            try:
                self.client.close()
                logger.info('Redis connection closed')
            except Exception as e:
                logger.warning(f'Error closing Redis connection: {e}')
            finally:
                self.client = None
