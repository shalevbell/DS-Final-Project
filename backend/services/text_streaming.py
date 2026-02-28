"""
Generic text streaming service for WebSocket communication.

Provides a reusable function to stream arbitrary text to the frontend
via WebSocket, independent of model-specific logic.
"""

import logging
from datetime import datetime
from typing import Dict, Optional
import eventlet

logger = logging.getLogger(__name__)


def stream_text(
    socketio,
    text: str,
    session_id: Optional[str] = None,
    metadata: Optional[Dict] = None
):
    """
    Stream arbitrary text to frontend via WebSocket.

    This function provides a generic way to stream any text content to the
    frontend, with optional context information. The text will be displayed
    using the frontend's TextStreamer class with letter-by-letter animation.

    Args:
        socketio: SocketIO instance for emitting events
        text: Text content to stream (can be any string)
        session_id: Optional session identifier for tracking
        metadata: Optional metadata dictionary for context
                 (e.g., {'source': 'whisper', 'chunk': 5, 'model': 'gpt-4'})

    Example:
        # Stream a simple message
        stream_text(socketio, "Processing complete!")

        # Stream with context
        stream_text(
            socketio,
            "Transcription: Hello world",
            session_id="session_123",
            metadata={'source': 'whisper', 'chunk': 5}
        )
    """
    try:
        payload = {
            'text': text,
            'timestamp': datetime.utcnow().isoformat()
        }

        if session_id:
            payload['sessionId'] = session_id

        if metadata:
            payload['metadata'] = metadata

        # Use eventlet.spawn for thread-safe emission from worker threads
        # This ensures the emission happens on the main greenthread
        eventlet.spawn(lambda: socketio.emit('text_stream', payload))

        logger.debug(
            f'[TextStream] Emitted text ({len(text)} chars) '
            f'session={session_id}, metadata={metadata}'
        )

    except Exception as e:
        logger.error(f'[TextStream] Error streaming text: {e}')
