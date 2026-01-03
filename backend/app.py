import logging
from flask import Flask, jsonify
from flask_cors import CORS
import psycopg2
import redis
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Enable CORS
CORS(app, origins=Config.CORS_ORIGINS)


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


@app.route('/', methods=['GET'])
def index():
    """Root endpoint."""
    return jsonify({
        'message': 'AI-Assisted Interview Analysis System API',
        'health_check': '/api/health'
    })


if __name__ == '__main__':
    logger.info(f"Starting Flask app on port 5000 (ENV: {Config.FLASK_ENV})")
    app.run(host='0.0.0.0', port=5000, debug=Config.DEBUG)
