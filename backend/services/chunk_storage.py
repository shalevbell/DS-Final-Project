"""
Chunk storage module.

Handles Redis storage operations and PUBSUB notifications for video/audio chunks.
"""

import logging
import json
from typing import Dict

from utils.redis_helper import (
    get_chunk_video_key,
    get_chunk_audio_key,
    get_chunk_meta_key,
    get_chunk_status_key,
    get_session_chunks_key
)

logger = logging.getLogger(__name__)


def store_chunk_in_redis(
    redis_client,
    session_id: str,
    chunk_index: int,
    video_bytes: bytes,
    audio_bytes: bytes,
    metadata: Dict
) -> int:
    """
    Store chunk data in Redis using pipeline for efficiency.

    Stores:
    - Video bytes
    - Audio bytes
    - Metadata (JSON)
    - Processing status ('pending')
    - Adds chunk index to session's chunk list

    Args:
        redis_client: Redis client instance
        session_id: Session identifier
        chunk_index: Chunk index within session
        video_bytes: MP4 video data
        audio_bytes: WAV audio data
        metadata: Dictionary with chunk metadata (timestamp, mimeType, durationMs, etc.)

    Returns:
        Total number of chunks for this session

    Raises:
        Exception if Redis operations fail
    """
    pipe = redis_client.pipeline()

    # Store chunk data with 1-hour TTL
    pipe.setex(get_chunk_video_key(session_id, chunk_index), 3600, video_bytes)
    pipe.setex(get_chunk_audio_key(session_id, chunk_index), 3600, audio_bytes)
    pipe.setex(get_chunk_meta_key(session_id, chunk_index), 3600, json.dumps(metadata))
    pipe.setex(get_chunk_status_key(session_id, chunk_index), 3600, 'pending')

    # Add chunk to session's chunk list
    chunks_key = get_session_chunks_key(session_id)
    pipe.rpush(chunks_key, chunk_index)
    pipe.expire(chunks_key, 3600)

    pipe.execute()

    # Get total chunk count
    chunk_count = redis_client.llen(chunks_key)

    logger.info(f'Stored chunk {chunk_index} for session {session_id} (total: {chunk_count})')

    return chunk_count


def notify_chunk_ready(
    redis_client,
    session_id: str,
    chunk_index: int,
    timestamp: int,
    video_size: int,
    audio_size: int,
    pubsub_channel: str
) -> None:
    """
    Publish PUBSUB notification that chunk is ready for processing.

    Args:
        redis_client: Redis client instance
        session_id: Session identifier
        chunk_index: Chunk index within session
        timestamp: Unix timestamp (milliseconds)
        video_size: Size of video data in bytes
        audio_size: Size of audio data in bytes
        pubsub_channel: PUBSUB channel name to publish to

    Raises:
        Exception if Redis PUBSUB publish fails
    """
    notification = {
        'sessionId': session_id,
        'chunkIndex': chunk_index,
        'timestamp': timestamp,
        'videoSize': video_size,
        'audioSize': audio_size
    }

    redis_client.publish(pubsub_channel, json.dumps(notification))

    logger.info(f'Published chunk notification: {session_id}:{chunk_index}')
