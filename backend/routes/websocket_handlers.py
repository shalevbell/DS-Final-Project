"""
WebSocket handlers module.

Registers SocketIO event handlers for real-time communication.
"""

import logging
import time
import json
import shutil
from flask_socketio import SocketIO, emit
import eventlet

from config import Config
from services.connection_manager import get_redis_client, initialize_chunk_processor, get_chunk_processor
from services.video_processor import parse_chunk_data, convert_chunk_with_ffmpeg
from services.chunk_storage import store_chunk_in_redis, notify_chunk_ready

logger = logging.getLogger(__name__)


def register_socketio_handlers(socketio: SocketIO, config: Config):
    """
    Register all SocketIO event handlers.

    Args:
        socketio: SocketIO instance
        config: Application configuration
    """

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection and ensure processor is started."""
        logger.info('Client connected')

        # Ensure chunk processor is started (for gunicorn deployment)
        processor = get_chunk_processor()
        if processor is None:
            eventlet.spawn_n(initialize_chunk_processor)

        emit('connected', {'status': 'connected', 'message': 'WebSocket connection established'})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        logger.info('Client disconnected')

    @socketio.on('request_camera_permission')
    def handle_camera_permission_request():
        """Handle camera permission request."""
        emit('camera_permission_requested', {
            'status': 'requested',
            'message': 'Please grant camera and microphone permissions'
        })

    @socketio.on('stream_ready')
    def handle_stream_ready(data):
        """Handle stream ready event and initialize storage."""
        if isinstance(data, dict):
            session_id = data.get('sessionId')
            logger.info(f'Stream ready: {session_id}')

            # Initialize stream storage in Redis
            if session_id:
                r = get_redis_client()
                if r:
                    try:
                        metadata = {
                            'sessionId': session_id,
                            'startTime': data.get('timestamp', int(time.time())),
                            'video': data.get('video'),
                            'audio': data.get('audio'),
                            'status': 'active'
                        }
                        r.setex(f'session:{session_id}:info', 3600, json.dumps(metadata))
                    except Exception as e:
                        logger.error(f'Error storing metadata: {e}')

            emit('stream_acknowledged', {
                'status': 'ready',
                'sessionId': session_id,
                'chunkDurationMs': config.CHUNK_DURATION_MS
            })

    @socketio.on('camera_error')
    def handle_camera_error(data):
        """Handle camera error event."""
        if isinstance(data, dict):
            logger.error(f'Camera error: {data.get("error", "Unknown")}')

    @socketio.on('video_chunk')
    def handle_video_chunk(data):
        """Handle video chunk from client and process through pipeline."""
        if not isinstance(data, dict):
            return

        session_id = data.get('sessionId')
        chunk_data = data.get('chunk')
        timestamp = data.get('timestamp', int(time.time()))
        chunk_index = data.get('chunkIndex')
        mime_type = data.get('mimeType')
        duration_ms = data.get('durationMs')

        if not session_id or not chunk_data:
            logger.warning('Invalid chunk data')
            return

        r = get_redis_client()
        if not r:
            logger.error('Redis not available')
            return

        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            logger.error('ffmpeg not available')
            return

        chunk_label = chunk_index if chunk_index is not None else int(timestamp)
        logger.info(f'Processing chunk {chunk_label}, size: {len(chunk_data)} bytes')

        try:
            # Step 1: Parse chunk data to normalized bytes
            input_bytes = parse_chunk_data(chunk_data)

            # Step 2: Convert using FFmpeg
            video_bytes, audio_bytes = convert_chunk_with_ffmpeg(input_bytes, ffmpeg_path)

            # Step 3: Store in Redis
            metadata = {
                'timestamp': timestamp,
                'chunkIndex': chunk_label,
                'mimeType': mime_type,
                'durationMs': duration_ms
            }
            store_chunk_in_redis(
                redis_client=r,
                session_id=session_id,
                chunk_index=chunk_label,
                video_bytes=video_bytes,
                audio_bytes=audio_bytes,
                metadata=metadata
            )

            # Step 4: Publish PUBSUB notification
            notify_chunk_ready(
                redis_client=r,
                session_id=session_id,
                chunk_index=chunk_label,
                timestamp=timestamp,
                video_size=len(video_bytes),
                audio_size=len(audio_bytes),
                pubsub_channel=config.PUBSUB_CHANNEL
            )

        except ValueError as e:
            logger.error(f'Chunk data parsing error: {e}')
        except Exception as e:
            logger.error(f'Error processing chunk: {e}')

    logger.info('SocketIO handlers registered')
