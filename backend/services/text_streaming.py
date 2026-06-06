"""
Generic text streaming service for WebSocket communication.

Provides a reusable function to stream arbitrary text to the frontend
via WebSocket, independent of model-specific logic.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

import eventlet.hubs

logger = logging.getLogger(__name__)


def stream_text(
    socketio,
    text: str,
    session_id: Optional[str] = None,
    metadata: Optional[Dict] = None
):
    try:
        payload = {
            'text': text,
            'timestamp': datetime.utcnow().isoformat()
        }

        if session_id:
            payload['sessionId'] = session_id

        if metadata:
            payload['metadata'] = metadata

        # schedule_call_global dispatches the emit onto the eventlet hub's OS
        # thread, which is safe to call from any thread including tpool workers.
        eventlet.hubs.get_hub().schedule_call_global(
            0, socketio.emit, 'text_stream', payload
        )

        logger.debug(
            f'[TextStream] Emitted text ({len(text)} chars) '
            f'session={session_id}, metadata={metadata}'
        )

    except Exception as e:
        logger.error(f'[TextStream] Error streaming text: {e}')
