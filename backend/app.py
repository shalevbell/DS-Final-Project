import logging
import os
import time
import json
import base64
import shutil
import subprocess
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import psycopg2
import redis
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
# Increase buffer size for large 30s video chunks.
socketio = SocketIO(
    app,
    cors_allowed_origins=Config.CORS_ORIGINS,
    async_mode='eventlet',
    max_http_buffer_size=100 * 1024 * 1024
)

# Configuration
CHUNK_DURATION_MS = 30000  # Video chunk duration in milliseconds (default: 30 seconds)

# Initialize Redis connection (lazy - will connect when needed)
redis_client = None
TMP_DIR = os.path.join(os.path.dirname(__file__), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

def get_redis():
    """Get Redis client, connecting if needed."""
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.from_url(Config.REDIS_URL, decode_responses=False)
            redis_client.ping()
            logger.info('Redis connected')
        except Exception as e:
            logger.error(f'Redis connection failed: {e}')
            redis_client = None
    return redis_client


def test_postgres_connection():
    """Test PostgreSQL connection."""
    try:
        conn = psycopg2.connect(Config.DATABASE_URL)
        conn.close()
        return True, "Connected"
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        return False, str(e)


def test_redis_connection():
    """Test Redis connection."""
    try:
        r = redis.from_url(Config.REDIS_URL)
        r.ping()
        return True, "Connected"
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return False, str(e)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint that tests all service connections."""
    postgres_ok, postgres_msg = test_postgres_connection()
    redis_ok, redis_msg = test_redis_connection()

    all_ok = postgres_ok and redis_ok
    status_code = 200 if all_ok else 503

    response = {
        'status': 'healthy' if all_ok else 'unhealthy',
        'flask': 'running',
        'services': {
            'postgresql': {
                'status': 'connected' if postgres_ok else 'disconnected',
                'message': postgres_msg
            },
            'redis': {
                'status': 'connected' if redis_ok else 'disconnected',
                'message': redis_msg
            }
        }
    }

    return jsonify(response), status_code


@app.route('/')
def index():
    """Serve the frontend index.html."""
    return send_from_directory(frontend_dir, 'index.html')


@app.route('/frontend/<path:path>')
def serve_frontend(path):
    """Serve frontend static files (CSS, JS)."""
    return send_from_directory(frontend_dir, path)


# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info('Client connected')
    emit('connected', {'status': 'connected', 'message': 'WebSocket connection established'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Client disconnected')


@socketio.on('request_camera_permission')
def handle_camera_permission_request():
    emit('camera_permission_requested', {'status': 'requested', 'message': 'Please grant camera and microphone permissions'})


@socketio.on('stream_ready')
def handle_stream_ready(data):
    if isinstance(data, dict):
        session_id = data.get('sessionId')
        logger.info(f'Stream ready: {session_id}')

        # Initialize stream storage in Redis
        if session_id:
            r = get_redis()
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
            'chunkDurationMs': CHUNK_DURATION_MS
        })


@socketio.on('camera_error')
def handle_camera_error(data):
    if isinstance(data, dict):
        logger.error(f'Camera error: {data.get("error", "Unknown")}')


@socketio.on('video_chunk')
def handle_video_chunk(data):
    """Handle video chunk from client and store in Redis."""
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

    r = get_redis()
    if not r:
        logger.error('Redis not available')
        return

    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        logger.error('ffmpeg not available')
        return

    # Handle chunk data - Socket.IO sends ArrayBuffer as dict/list
    if isinstance(chunk_data, bytes):
        input_bytes = chunk_data
    elif isinstance(chunk_data, (list, bytearray)):
        input_bytes = bytes(chunk_data)
    elif isinstance(chunk_data, dict):
        # Socket.IO might wrap ArrayBuffer in {'data': [...], 'type': 'Buffer'}
        if 'data' in chunk_data:
            input_bytes = bytes(chunk_data['data'])
        else:
            logger.error(f'Unexpected dict format: {list(chunk_data.keys())}')
            return
    elif isinstance(chunk_data, str):
        # Legacy base64 support
        try:
            input_bytes = base64.b64decode(chunk_data)
        except Exception as e:
            logger.error(f'Failed to decode base64: {e}')
            return
    else:
        logger.error(f'Unexpected chunk data type: {type(chunk_data).__name__}')
        return

    chunk_label = chunk_index if chunk_index is not None else int(timestamp)
    logger.info(f'Processing chunk {chunk_label}, size: {len(input_bytes)} bytes')

    try:
        # Convert video using ffmpeg pipes (no temp files)
        # Use fragmented MP4 for pipe compatibility
        video_proc = subprocess.run(
            [
                ffmpeg_path, '-hide_banner', '-loglevel', 'error',
                '-i', 'pipe:0',
                '-c:v', 'libx264', '-preset', 'veryfast',
                '-pix_fmt', 'yuv420p',
                '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
                '-f', 'mp4', 'pipe:1'
            ],
            input=input_bytes,
            capture_output=True,
            check=True
        )
        video_bytes = video_proc.stdout

        # Convert audio using ffmpeg pipes (no temp files)
        audio_proc = subprocess.run(
            [
                ffmpeg_path, '-hide_banner', '-loglevel', 'error',
                '-i', 'pipe:0',
                '-vn',  # No video
                '-map', '0:a:0',  # Map first audio stream
                '-c:a', 'libmp3lame',  # MP3 codec
                '-b:a', '128k',  # Bitrate
                '-ar', '44100',  # Standard sample rate
                '-ac', '2',  # Stereo (2 channels)
                '-f', 'mp3', 'pipe:1'
            ],
            input=input_bytes,
            capture_output=True,
            check=True
        )
        audio_bytes = audio_proc.stdout

        # Store in Redis using pipeline for efficiency
        meta = {
            'timestamp': timestamp,
            'chunkIndex': chunk_label,
            'mimeType': mime_type,
            'durationMs': duration_ms
        }
        
        # REDIS SET
        pipe = r.pipeline()
        pipe.setex(f'session:{session_id}:chunk:{chunk_label}:video', 3600, video_bytes)
        pipe.setex(f'session:{session_id}:chunk:{chunk_label}:audio', 3600, audio_bytes)
        pipe.setex(f'session:{session_id}:chunk:{chunk_label}:meta', 3600, json.dumps(meta))
        pipe.rpush(f'session:{session_id}:chunks', chunk_label)
        pipe.expire(f'session:{session_id}:chunks', 3600)
        pipe.execute()

        chunk_count = r.llen(f'session:{session_id}:chunks')
        logger.info(f'Stored chunk {chunk_label} for session {session_id} (total: {chunk_count})')

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else 'no error output'
        logger.error(f'ffmpeg failed (code {e.returncode}): {stderr_output[:200]}')
    except Exception as e:
        logger.error(f'Error processing chunk: {e}')


if __name__ == '__main__':
    logger.info(f"Starting Flask app with SocketIO on port 5000 (ENV: {Config.FLASK_ENV})")
    socketio.run(app, host='0.0.0.0', port=5000, debug=Config.DEBUG)
