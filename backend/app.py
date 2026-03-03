"""
Main Flask application with SocketIO support.

Initializes the Flask app, registers routes/handlers, and starts the chunk processor.
"""

import logging
import os
import signal
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import eventlet

from config import Config
from services.connection_manager import (
    get_redis_client,
    initialize_chunk_processor,
    create_shutdown_handler
)
from routes.http_routes import register_http_routes
from routes.websocket_handlers import register_socketio_handlers
from services.text_streaming import stream_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s]-[%(asctime)s]-[%(name)s]: %(message)s'
)
logger = logging.getLogger(__name__)

# Get the frontend directory path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
frontend_dir = os.path.join(project_root, 'frontend')

if not os.path.exists(frontend_dir):
    frontend_dir = '/frontend'

logger.info(f"Frontend directory: {frontend_dir}")

# Initialize Flask app with frontend directory configuration
app = Flask(__name__,
            static_folder=frontend_dir,
            static_url_path='',
            template_folder=frontend_dir)
app.config.from_object(Config)

# Enable CORS
CORS(app, origins=Config.CORS_ORIGINS)

# Initialize SocketIO
# Increase buffer size for large 30s video chunks
socketio = SocketIO(
    app,
    cors_allowed_origins=Config.CORS_ORIGINS,
    async_mode='eventlet',
    max_http_buffer_size=100 * 1024 * 1024
)

# Preload ML models before starting chunk processor
from services.model_loader import preload_all_models
model_status = preload_all_models()

# Initialize services
redis_client = get_redis_client()
chunk_processor = initialize_chunk_processor(socketio_instance=socketio)

# Register routes and handlers
register_http_routes(app, chunk_processor, frontend_dir)
register_socketio_handlers(socketio, Config)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, create_shutdown_handler(chunk_processor))
signal.signal(signal.SIGTERM, create_shutdown_handler(chunk_processor))


def get_text_streamer():
    """
    Returns a partial function bound to the global socketio instance.

    This allows any module to stream text without needing direct access
    to the socketio instance.

    Usage:
        from app import get_text_streamer
        streamer = get_text_streamer()
        streamer("Hello from anywhere!", session_id="session-123")

    Returns:
        Partial function with socketio already bound
    """
    from functools import partial
    return partial(stream_text, socketio)

if __name__ == '__main__':
    logger.info(f"Starting Flask app with SocketIO on port 5000 (ENV: {Config.FLASK_ENV})")
    socketio.run(app, host='0.0.0.0', port=5000, debug=Config.DEBUG)
