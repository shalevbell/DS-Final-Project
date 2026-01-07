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
            logger.info('✅ Redis connected')
        except Exception as e:
            logger.error(f'❌ Redis connection failed: {e}')
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
def handle_camera_permission_request(data=None):
    logger.info('Camera permission requested')
    emit('camera_permission_requested', {'status': 'requested', 'message': 'Please grant camera and microphone permissions'})


@socketio.on('camera_status')
def handle_camera_status(data):
    if isinstance(data, dict):
        logger.info(f'Camera status: {data.get("status", "unknown")}')
        emit('camera_status_ack', {'status': 'received', 'data': data})


@socketio.on('stream_ready')
def handle_stream_ready(data):
    if isinstance(data, dict):
        session_id = data.get('sessionId')
        logger.info(f'Stream ready: session={session_id}, video={data.get("video")}, audio={data.get("audio")}')
        
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
                    r.setex(f'stream:{session_id}:metadata', 3600, json.dumps(metadata))
                    logger.info(f'✅ Stream metadata stored for session {session_id}')
                except Exception as e:
                    logger.error(f'❌ Error storing metadata: {e}')
        
        emit('stream_acknowledged', {'status': 'ready', 'sessionId': session_id})


@socketio.on('camera_error')
def handle_camera_error(data):
    if isinstance(data, dict):
        logger.error(f'Camera error: {data.get("error", "Unknown")} - {data.get("message", "")}')


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
        logger.warning('Invalid video chunk data received')
        return
    
    r = get_redis()
    if not r:
        logger.error('❌ Redis not available, cannot store chunk')
        return
    
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        logger.error('❌ ffmpeg not available, cannot convert chunk to mp4/mp3')
        return

    input_bytes = None
    if isinstance(chunk_data, (bytes, bytearray, memoryview)):
        input_bytes = bytes(chunk_data)
    elif isinstance(chunk_data, list):
        input_bytes = bytes(chunk_data)
    elif isinstance(chunk_data, dict) and isinstance(chunk_data.get('data'), list):
        input_bytes = bytes(chunk_data['data'])
    else:
        if isinstance(chunk_data, bytes):
            chunk_data = chunk_data.decode('utf-8', errors='ignore')
        if isinstance(chunk_data, str):
            chunk_data = chunk_data.strip()
            # Fix missing base64 padding from some browsers.
            padding = len(chunk_data) % 4
            if padding:
                chunk_data += '=' * (4 - padding)
        try:
            input_bytes = base64.b64decode(chunk_data)
        except Exception as e:
            logger.error(f'❌ Failed to decode chunk base64: {e}')
            return

    chunk_label = chunk_index if chunk_index is not None else int(timestamp)
    input_path = os.path.join(TMP_DIR, f'{session_id}_{chunk_label}.webm')
    output_video_path = os.path.join(TMP_DIR, f'{session_id}_{chunk_label}.mp4')
    output_audio_path = os.path.join(TMP_DIR, f'{session_id}_{chunk_label}.mp3')

    try:
        with open(input_path, 'wb') as f:
            f.write(input_bytes)

        video_proc = subprocess.run(
            [
                ffmpeg_path, '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'veryfast',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                output_video_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        audio_proc = subprocess.run(
            [
                ffmpeg_path, '-y',
                '-i', input_path,
                '-vn',
                '-c:a', 'libmp3lame',
                '-b:a', '128k',
                output_audio_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        with open(output_video_path, 'rb') as f:
            video_bytes = f.read()
        with open(output_audio_path, 'rb') as f:
            audio_bytes = f.read()

        base_key = f'stream:{session_id}:chunk:{chunk_label}'
        r.setex(f'{base_key}:video:mp4', 3600, video_bytes)
        r.setex(f'{base_key}:audio:mp3', 3600, audio_bytes)

        meta = {
            'timestamp': timestamp,
            'chunkIndex': chunk_label,
            'mimeType': mime_type,
            'durationMs': duration_ms
        }
        r.setex(f'{base_key}:meta', 3600, json.dumps(meta))
        r.rpush(f'stream:{session_id}:chunks', chunk_label)
        r.expire(f'stream:{session_id}:chunks', 3600)

        r.setex(f'stream:{session_id}:last_chunk', 60, timestamp)
        chunk_count = r.llen(f'stream:{session_id}:chunks')
        logger.info(
            f'✅ Stored mp4/mp3 chunk for session {session_id} - chunk #{chunk_label}, '
            f'durationMs={duration_ms}, total: {chunk_count}'
        )
    except subprocess.CalledProcessError as e:
        stderr = ''
        if e.stderr:
            stderr = e.stderr.strip()
        logger.error(f'❌ ffmpeg failed to convert chunk to mp4/mp3: {stderr or "no stderr"}')
    except Exception as e:
        logger.error(f'❌ Error processing video chunk: {e}')
    finally:
        for path in (input_path, output_video_path, output_audio_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


@socketio.on('test_redis')
def handle_test_redis(data):
    """Test Redis connection."""
    logger.info('🧪 Testing Redis connection...')
    
    try:
        r = get_redis()
        if r:
            # Test basic operations
            test_key = 'test:connection'
            r.setex(test_key, 10, 'test_value')
            value = r.get(test_key)
            r.delete(test_key)
            
            logger.info('✅ Redis connection test successful!')
            emit('redis_test_result', {
                'status': 'success',
                'message': 'Redis connection working',
                'test_value': value.decode() if isinstance(value, bytes) else value
            })
        else:
            logger.error('❌ Redis connection test failed - client is None')
            emit('redis_test_result', {
                'status': 'failed',
                'message': 'Redis client not available'
            })
    except Exception as e:
        logger.error(f'❌ Redis connection test error: {e}')
        emit('redis_test_result', {
            'status': 'error',
            'message': str(e)
        })


if __name__ == '__main__':
    logger.info(f"Starting Flask app with SocketIO on port 5000 (ENV: {Config.FLASK_ENV})")
    socketio.run(app, host='0.0.0.0', port=5000, debug=Config.DEBUG)
