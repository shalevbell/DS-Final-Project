"""
HTTP routes module.

Registers HTTP endpoints for health checks and frontend serving.
"""

import logging
from flask import Flask, jsonify, request, send_from_directory
from chunk_processor import ChunkProcessor
from services.connection_manager import check_postgres_health, check_redis_health
from services.model_loader import get_preload_status
from services.db_service import list_sessions, get_session_with_chunks, get_chunk_detail
from run_models import list_savee_dataset_files

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

    @app.route('/api/models/status', methods=['GET'])
    def models_status():
        """
        Return model preloading status.

        Returns:
            JSON with status of each model:
            {
                "whisper": {"ready": bool, "loading": bool, "error": str|null},
                "mediapipe": {"ready": bool, "loading": bool, "error": str|null},
                "vocaltone": {"ready": bool, "loading": bool, "error": str|null},
                "ollama": {"ready": bool, "loading": bool, "error": str|null},
                "all_ready": bool
            }
        """
        status = get_preload_status()
        all_ready = all(model['ready'] for model in status.values())

        return jsonify({
            **status,
            'all_ready': all_ready
        }), 200

    @app.route('/api/vocal-tone/dataset/list', methods=['GET'])
    def list_vocal_tone_dataset():
        """
        List all files in the SAVEE dataset directory for Vocal Tone model training.
        
        Returns:
            JSON with file listing results including file names, sizes, and extensions.
        """
        result = list_savee_dataset_files()
        
        if result.get('error'):
            return jsonify(result), 404
        
        return jsonify(result), 200

    def _serialize(obj):
        """Convert datetime objects to ISO strings for JSON serialization."""
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return obj

    def _serialize_row(row: dict) -> dict:
        return {k: _serialize(v) for k, v in row.items()}

    @app.route('/api/history/sessions', methods=['GET'])
    def history_sessions():
        """List sessions with optional candidate name filter and pagination."""
        candidate = request.args.get('candidate', '').strip() or None
        try:
            limit = min(int(request.args.get('limit', 20)), 200)
            offset = max(int(request.args.get('offset', 0)), 0)
        except ValueError:
            limit, offset = 20, 0

        result = list_sessions(candidate_filter=candidate, limit=limit, offset=offset)
        if result is None:
            return jsonify({'error': 'Database unavailable'}), 500

        sessions, total = result
        return jsonify({
            'sessions': [_serialize_row(s) for s in sessions],
            'total': total,
            'limit': limit,
            'offset': offset,
        }), 200

    @app.route('/api/history/sessions/<session_id>', methods=['GET'])
    def history_session_detail(session_id):
        """Return a session and all its chunk results."""
        data = get_session_with_chunks(session_id)
        if data is None:
            return jsonify({'error': 'Session not found'}), 404
        return jsonify({
            'session': _serialize_row(data['session']),
            'chunks': [_serialize_row(c) for c in data['chunks']],
        }), 200

    @app.route('/api/history/sessions/<session_id>/chunks/<int:chunk_index>', methods=['GET'])
    def history_chunk_detail(session_id, chunk_index):
        """Return a single chunk result."""
        chunk = get_chunk_detail(session_id, chunk_index)
        if chunk is None:
            return jsonify({'error': 'Chunk not found'}), 404
        return jsonify({'chunk': _serialize_row(chunk)}), 200

    @app.route('/api/sessions/<session_id>/complete', methods=['POST'])
    def complete_session_endpoint(session_id):
        """Complete a session — called by sendBeacon on page unload."""
        try:
            from services.db_service import complete_session
            complete_session(session_id)
        except Exception as e:
            logger.warning(f'complete_session endpoint failed for {session_id}: {e}')
        return '', 204

    @app.route('/history')
    def history_page():
        """Serve the history browser page."""
        return send_from_directory(frontend_dir, 'history.html')

    @app.route('/')
    def index():
        """Serve the frontend index.html."""
        return send_from_directory(frontend_dir, 'index.html')

    @app.route('/frontend/<path:path>')
    def serve_frontend(path):
        """Serve frontend static files (CSS, JS)."""
        return send_from_directory(frontend_dir, path)

    logger.info('HTTP routes registered')
