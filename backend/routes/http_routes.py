"""
HTTP routes module.

Registers HTTP endpoints for health checks and frontend serving.
"""

import logging
from flask import Flask, jsonify, send_from_directory
from chunk_processor import ChunkProcessor
from services.connection_manager import check_postgres_health, check_redis_health

logger = logging.getLogger(__name__)


def register_http_routes(app: Flask, chunk_processor: ChunkProcessor, frontend_dir: str):
    """
    Register all HTTP routes with the Flask app.

    Args:
        app: Flask application instance
        chunk_processor: ChunkProcessor instance for health checks
        frontend_dir: Path to frontend directory for serving static files
    """

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint that tests all service connections."""
        postgres_ok, postgres_msg = check_postgres_health()
        redis_ok, redis_msg = check_redis_health()

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

    @app.route('/api/health/processing', methods=['GET'])
    def processing_health():
        """Return processing queue statistics."""
        if chunk_processor:
            stats = chunk_processor.get_stats()
            return jsonify({
                'status': 'running',
                'queue_size': stats['queue_size'],
                'in_progress': stats['in_progress'],
                'completed_total': stats['completed_total'],
                'failed_total': stats['failed_total'],
                'workers': stats['workers'],
                'chunks_received': stats['chunks_received'],
                'chunks_processed': stats['chunks_processed'],
                'avg_processing_time_ms': stats['avg_processing_time_ms']
            }), 200
        return jsonify({'error': 'Processor not initialized'}), 503

    @app.route('/')
    def index():
        """Serve the frontend index.html."""
        return send_from_directory(frontend_dir, 'index.html')

    @app.route('/frontend/<path:path>')
    def serve_frontend(path):
        """Serve frontend static files (CSS, JS)."""
        return send_from_directory(frontend_dir, path)

    logger.info('HTTP routes registered')
