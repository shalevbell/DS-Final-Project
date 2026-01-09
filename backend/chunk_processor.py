"""
Chunk processor with Redis PUBSUB subscription.

Manages background processing of video/audio chunks using parallel analysis.
"""

import logging
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict

import redis
import eventlet

from processing_queue import ProcessingQueue
from analysis_functions import run_parallel_analysis, get_available_models
from utils.redis_helper import (
    get_chunk_video_key,
    get_chunk_audio_key,
    get_chunk_meta_key,
    get_chunk_status_key,
    get_chunk_results_key,
    get_chunk_error_key
)

logger = logging.getLogger(__name__)


class ChunkProcessor:
    """
    Main processor that manages PUBSUB subscription and chunk processing.

    Runs in background thread alongside Flask app using eventlet greenthreads.
    Processes chunks in parallel using ThreadPoolExecutor.
    """

    def __init__(
        self,
        redis_url: str,
        max_workers: int = 3,
        queue_size: int = 100,
        pubsub_channel: str = 'chunks:ready',
        reconnect_delay: int = 5
    ):
        """
        Initialize chunk processor.

        Args:
            redis_url: Redis connection URL
            max_workers: Maximum parallel workers for chunk processing
            queue_size: Maximum queue size
            pubsub_channel: PUBSUB channel name
            reconnect_delay: Seconds to wait before reconnection attempt
        """
        self.redis_url = redis_url
        self.max_workers = max_workers
        self.pubsub_channel = pubsub_channel
        self.reconnect_delay = reconnect_delay

        # Create separate Redis clients (PUBSUB requires dedicated connection)
        self.redis_pubsub_client: Optional[redis.Redis] = None
        self.redis_data_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None

        # Initialize processing queue
        self.queue = ProcessingQueue(maxsize=queue_size)

        # Initialize thread pool for chunk processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Shutdown event
        self.shutdown_event = threading.Event()

        # Metrics
        self.metrics = {
            'chunks_received': 0,
            'chunks_processed': 0,
            'chunks_failed': 0,
            'processing_times': []
        }
        self.metrics_lock = threading.Lock()

        logger.info(
            f'ChunkProcessor initialized: workers={max_workers}, '
            f'queue={queue_size}, channel={pubsub_channel}'
        )

    def _connect_redis(self):
        """Initialize Redis connections."""
        try:
            # PUBSUB connection (dedicated)
            self.redis_pubsub_client = redis.from_url(
                self.redis_url,
                decode_responses=True  # Decode for message parsing
            )
            self.redis_pubsub_client.ping()

            # Data connection (for retrieving/storing chunks)
            self.redis_data_client = redis.from_url(
                self.redis_url,
                decode_responses=False  # Binary mode for video/audio
            )
            self.redis_data_client.ping()

            logger.info('Redis connections established for chunk processor')

        except Exception as e:
            logger.error(f'Redis connection failed: {e}')
            raise

    def start(self):
        """Start background PUBSUB listener and queue processor."""
        # Connect to Redis
        self._connect_redis()

        # Start PUBSUB listener in eventlet greenthread
        eventlet.spawn(self._pubsub_listener)

        # Start queue processor in eventlet greenthread
        eventlet.spawn(self._queue_processor)

        logger.info('ChunkProcessor started (PUBSUB listener + queue processor)')

    def _pubsub_listener(self):
        """
        Subscribe to Redis PUBSUB and add messages to queue.

        Runs in eventlet greenthread. Handles reconnection on failure.
        """
        logger.info(f'PUBSUB listener started, subscribing to "{self.pubsub_channel}"')

        while not self.shutdown_event.is_set():
            try:
                # Create PUBSUB subscription
                self.pubsub = self.redis_pubsub_client.pubsub()
                self.pubsub.subscribe(self.pubsub_channel)

                logger.info(f'Subscribed to channel: {self.pubsub_channel}')

                # Listen for messages
                for message in self.pubsub.listen():
                    if self.shutdown_event.is_set():
                        break

                    if message['type'] == 'message':
                        self._handle_pubsub_message(message)

            except Exception as e:
                logger.error(f'PUBSUB listener error: {e}')

                if not self.shutdown_event.is_set():
                    logger.info(f'Reconnecting in {self.reconnect_delay}s...')
                    eventlet.sleep(self.reconnect_delay)

                    # Recreate connection
                    try:
                        self._connect_redis()
                    except Exception as reconnect_error:
                        logger.error(f'Reconnection failed: {reconnect_error}')

        logger.info('PUBSUB listener stopped')

    def _handle_pubsub_message(self, message: Dict):
        """
        Parse PUBSUB message and add to processing queue.

        Args:
            message: PUBSUB message dict
        """
        try:
            data = json.loads(message['data'])
            session_id = data.get('sessionId')
            chunk_index = data.get('chunkIndex')
            timestamp = data.get('timestamp')

            if not session_id or chunk_index is None:
                logger.warning(f'Invalid PUBSUB message: {data}')
                return

            logger.info(
                f'Received chunk notification: {session_id}:{chunk_index} '
                f'(video={data.get("videoSize", 0)} bytes, audio={data.get("audioSize", 0)} bytes)'
            )

            # Add to queue
            success = self.queue.add(session_id, chunk_index)

            if success:
                with self.metrics_lock:
                    self.metrics['chunks_received'] += 1
            else:
                logger.debug(f'Chunk {session_id}:{chunk_index} not added (duplicate or queue full)')

        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse PUBSUB message: {e}')
        except Exception as e:
            logger.error(f'Error handling PUBSUB message: {e}')

    def _queue_processor(self):
        """
        Process chunks from queue using thread pool.

        Runs in eventlet greenthread. Submits chunks to ThreadPoolExecutor.
        """
        logger.info('Queue processor started')

        while not self.shutdown_event.is_set():
            try:
                # Get next chunk from queue (1s timeout)
                result = self.queue.get(timeout=1.0)

                if result is None:
                    # Queue is empty, continue waiting
                    continue

                session_id, chunk_index = result

                # Submit to thread pool for processing
                self.executor.submit(self._process_chunk_with_retry, session_id, chunk_index)

            except Exception as e:
                logger.error(f'Queue processor error: {e}')
                eventlet.sleep(1.0)

        logger.info('Queue processor stopped')

    def _process_chunk_with_retry(
        self,
        session_id: str,
        chunk_index: int,
        attempt: int = 1,
        max_attempts: int = 3
    ):
        """
        Process chunk with retry logic.

        Args:
            session_id: Session identifier
            chunk_index: Chunk index within session
            attempt: Current attempt number
            max_attempts: Maximum retry attempts
        """
        try:
            self._process_chunk(session_id, chunk_index)

        except redis.ConnectionError as e:
            # Transient error - retry
            if attempt < max_attempts:
                delay = 5 * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(
                    f'Retrying chunk {session_id}:{chunk_index} after {delay}s '
                    f'(attempt {attempt}/{max_attempts}): {e}'
                )
                time.sleep(delay)
                self._process_chunk_with_retry(session_id, chunk_index, attempt + 1, max_attempts)
            else:
                logger.error(f'Max retries exceeded for chunk {session_id}:{chunk_index}')
                self._mark_failed(session_id, chunk_index, str(e))

        except Exception as e:
            # Permanent error - don't retry
            logger.error(f'Permanent error for chunk {session_id}:{chunk_index}: {e}')
            self._mark_failed(session_id, chunk_index, str(e))

        finally:
            # Always mark as complete in queue
            self.queue.mark_complete(session_id, chunk_index)

    def _process_chunk(self, session_id: str, chunk_index: int):
        """
        Process a single chunk with parallel analysis.

        Args:
            session_id: Session identifier
            chunk_index: Chunk index within session
        """
        start_time = time.time()

        logger.info(f'Processing chunk {session_id}:{chunk_index}')

        # 1. Update status to 'processing'
        self._set_status(session_id, chunk_index, 'processing')

        # 2. Retrieve chunk data from Redis
        video_bytes, audio_bytes, meta = self._get_chunk_data(session_id, chunk_index)

        if video_bytes is None or audio_bytes is None:
            raise ValueError(f'Missing chunk data in Redis for {session_id}:{chunk_index}')

        # 3. Run parallel analysis
        # RUNS THE MODELS FOR THE CHUNK
        results = run_parallel_analysis(
            video_bytes=video_bytes,
            audio_bytes=audio_bytes,
            session_id=session_id,
            chunk_index=chunk_index
        )

        # 4. Store results in Redis
        self._store_results(session_id, chunk_index, results)

        # 5. Update status to 'completed'
        self._set_status(session_id, chunk_index, 'completed')

        # 6. Update metrics
        processing_time = int((time.time() - start_time) * 1000)

        with self.metrics_lock:
            self.metrics['chunks_processed'] += 1
            self.metrics['processing_times'].append(processing_time)

            # Keep only last 100 processing times
            if len(self.metrics['processing_times']) > 100:
                self.metrics['processing_times'] = self.metrics['processing_times'][-100:]

        logger.info(f'Completed chunk {session_id}:{chunk_index} in {processing_time}ms')

    def _get_chunk_data(self, session_id: str, chunk_index: int):
        """
        Retrieve chunk data from Redis.

        Args:
            session_id: Session identifier
            chunk_index: Chunk index within session

        Returns:
            Tuple of (video_bytes, audio_bytes, meta_dict)
        """
        try:
            video_key = get_chunk_video_key(session_id, chunk_index)
            audio_key = get_chunk_audio_key(session_id, chunk_index)
            meta_key = get_chunk_meta_key(session_id, chunk_index)

            # Retrieve data
            video_bytes = self.redis_data_client.get(video_key)
            audio_bytes = self.redis_data_client.get(audio_key)
            meta_json = self.redis_data_client.get(meta_key)

            # Parse metadata
            meta = json.loads(meta_json.decode('utf-8')) if meta_json else {}

            return video_bytes, audio_bytes, meta

        except Exception as e:
            logger.error(f'Error retrieving chunk data for {session_id}:{chunk_index}: {e}')
            raise

    def _store_results(self, session_id: str, chunk_index: int, results: Dict):
        """
        Store analysis results in Redis.

        Args:
            session_id: Session identifier
            chunk_index: Chunk index within session
            results: Analysis results dictionary
        """
        try:
            pipe = self.redis_data_client.pipeline()

            # Store each model's results separately
            for model in get_available_models():
                if model in results:
                    key = get_chunk_results_key(session_id, chunk_index, model)
                    pipe.setex(key, 3600, json.dumps(results[model]))

            pipe.execute()

            logger.debug(f'Stored results for chunk {session_id}:{chunk_index}')

        except Exception as e:
            logger.error(f'Error storing results for {session_id}:{chunk_index}: {e}')
            raise

    def _set_status(self, session_id: str, chunk_index: int, status: str):
        """
        Update chunk processing status in Redis.

        Args:
            session_id: Session identifier
            chunk_index: Chunk index within session
            status: Status value ('pending', 'processing', 'completed', 'failed')
        """
        try:
            key = get_chunk_status_key(session_id, chunk_index)
            self.redis_data_client.setex(key, 3600, status)
        except Exception as e:
            logger.warning(f'Error setting status for {session_id}:{chunk_index}: {e}')

    def _mark_failed(self, session_id: str, chunk_index: int, error_message: str):
        """
        Mark chunk as failed and store error information.

        Args:
            session_id: Session identifier
            chunk_index: Chunk index within session
            error_message: Error description
        """
        try:
            # Update status
            self._set_status(session_id, chunk_index, 'failed')

            # Store error details
            error_data = {
                'error': error_message,
                'timestamp': int(time.time())
            }

            error_key = get_chunk_error_key(session_id, chunk_index)
            self.redis_data_client.setex(error_key, 3600, json.dumps(error_data))

            # Update metrics
            with self.metrics_lock:
                self.metrics['chunks_failed'] += 1

            logger.error(f'Marked chunk {session_id}:{chunk_index} as failed: {error_message}')

        except Exception as e:
            logger.error(f'Error marking chunk as failed: {e}')

    def get_stats(self) -> Dict:
        """
        Get current processor statistics.

        Returns:
            Dictionary with metrics and queue statistics
        """
        queue_stats = self.queue.size()

        with self.metrics_lock:
            processing_times = self.metrics['processing_times']
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0

            stats = {
                'workers': self.max_workers,
                'chunks_received': self.metrics['chunks_received'],
                'chunks_processed': self.metrics['chunks_processed'],
                'chunks_failed': self.metrics['chunks_failed'],
                'queue_size': queue_stats['queue_size'],
                'in_progress': queue_stats['in_progress'],
                'completed_total': self.metrics['chunks_processed'],
                'failed_total': self.metrics['chunks_failed'],
                'avg_processing_time_ms': int(avg_time)
            }

        return stats

    def shutdown(self):
        """
        Gracefully shutdown processor.

        Stops PUBSUB listener, drains queue, waits for active chunks, closes connections.
        """
        logger.info('Shutting down chunk processor...')

        # Signal shutdown
        self.shutdown_event.set()

        # Unsubscribe from PUBSUB
        if self.pubsub:
            try:
                self.pubsub.unsubscribe()
                self.pubsub.close()
            except Exception as e:
                logger.warning(f'Error closing PUBSUB: {e}')

        # Shutdown thread pool (wait for active chunks to complete)
        logger.info('Waiting for active chunks to complete...')
        self.executor.shutdown(wait=True, cancel_futures=False)

        # Close Redis connections
        if self.redis_pubsub_client:
            try:
                self.redis_pubsub_client.close()
            except Exception as e:
                logger.warning(f'Error closing Redis PUBSUB client: {e}')

        if self.redis_data_client:
            try:
                self.redis_data_client.close()
            except Exception as e:
                logger.warning(f'Error closing Redis data client: {e}')

        logger.info('Chunk processor shutdown complete')
