"""
Generic text streaming service for WebSocket communication.

Provides a reusable function to stream arbitrary text to the frontend
via WebSocket, independent of model-specific logic.
"""

import logging
from datetime import datetime
from typing import Dict, Optional
import eventlet.patcher as _ep

# Use the real (unpatched) Queue so OS threads can put() without deadlocking.
_real_queue = _ep.original('queue')
_emit_queue = _real_queue.Queue()

logger = logging.getLogger(__name__)


def start_emit_worker(socketio):
    """
    Drain _emit_queue in a greenthread on the main eventlet hub.
    Must be called once from app.py after socketio is created.
    """
    import eventlet

    def _worker():
        while True:
            try:
                payload = _emit_queue.get(timeout=1)
                socketio.emit('text_stream', payload)
            except _real_queue.Empty:
                eventlet.sleep(0)
            except Exception as e:
                logger.error(f'[TextStream] emit worker error: {e}')

    eventlet.spawn(_worker)


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

        # put() is safe from any OS thread; the greenthread worker emits it.
        _emit_queue.put(payload)

        logger.debug(
            f'[TextStream] Queued text ({len(text)} chars) '
            f'session={session_id}, metadata={metadata}'
        )

    except Exception as e:
        logger.error(f'[TextStream] Error queueing text: {e}')
