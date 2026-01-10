"""
Thread-safe processing queue for chunk processing.

Provides FIFO ordering, duplicate detection, and backpressure management.
"""

import logging
import queue
import threading
from typing import Optional, Tuple, Set

logger = logging.getLogger(__name__)


class ProcessingQueue:
    """
    Thread-safe queue for managing chunk processing with duplicate prevention.

    Features:
    - FIFO ordering
    - Duplicate detection (prevents same chunk from being queued/processed multiple times)
    - Backpressure warnings when queue fills up
    - Thread-safe operations
    """

    def __init__(self, maxsize: int = 100):
        """
        Initialize processing queue.

        Args:
            maxsize: Maximum queue size before blocking (default: 100)
        """
        self.queue = queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize

        # Track chunks currently being processed
        self.in_progress: Set[str] = set()

        # Track chunks in the queue (not yet started)
        self.pending: Set[str] = set()

        # Lock for thread-safe set operations
        self.lock = threading.Lock()

        logger.info(f'ProcessingQueue initialized with maxsize={maxsize}')

    def _make_key(self, session_id: str, chunk_index: int) -> str:
        """Create unique key for session:chunk combination."""
        return f"{session_id}:{chunk_index}"

    def add(self, session_id: str, chunk_index: int, timeout: float = 5.0) -> bool:
        """
        Add a chunk to the processing queue.

        Args:
            session_id: Session identifier
            chunk_index: Chunk index within session
            timeout: Seconds to wait if queue is full (default: 5.0)

        Returns:
            True if added successfully, False if duplicate or queue full
        """
        chunk_key = self._make_key(session_id, chunk_index)

        with self.lock:
            # Skip if already processing
            if chunk_key in self.in_progress:
                logger.debug(f'Chunk {chunk_key} already in progress, skipping')
                return False

            # Skip if already in queue
            if chunk_key in self.pending:
                logger.debug(f'Chunk {chunk_key} already queued, skipping')
                return False

            # Check backpressure
            queue_size = self.queue.qsize()
            if queue_size > self.maxsize * 0.9:
                logger.warning(
                    f'Queue near capacity: {queue_size}/{self.maxsize} '
                    f'({queue_size/self.maxsize*100:.1f}%)'
                )

            # Try to add to queue
            try:
                self.queue.put((session_id, chunk_index), block=True, timeout=timeout)
                self.pending.add(chunk_key)
                logger.debug(f'Added chunk {chunk_key} to queue (size: {queue_size + 1})')
                return True
            except queue.Full:
                logger.error(
                    f'Queue full, could not add chunk {chunk_key} within {timeout}s timeout'
                )
                return False

    def get(self, timeout: float = 1.0) -> Optional[Tuple[str, int]]:
        """
        Get the next chunk to process from the queue.

        Args:
            timeout: Seconds to wait for an item (default: 1.0)

        Returns:
            Tuple of (session_id, chunk_index) or None if queue is empty
        """
        try:
            session_id, chunk_index = self.queue.get(block=True, timeout=timeout)
            chunk_key = self._make_key(session_id, chunk_index)

            with self.lock:
                # Remove from pending, add to in-progress
                self.pending.discard(chunk_key)
                self.in_progress.add(chunk_key)

            logger.debug(f'Retrieved chunk {chunk_key} from queue')
            return session_id, chunk_index

        except queue.Empty:
            return None

    def mark_complete(self, session_id: str, chunk_index: int):
        """
        Mark a chunk as completed and remove from in-progress tracking.

        Args:
            session_id: Session identifier
            chunk_index: Chunk index within session
        """
        chunk_key = self._make_key(session_id, chunk_index)

        with self.lock:
            self.in_progress.discard(chunk_key)

        logger.debug(f'Marked chunk {chunk_key} as complete')

    def size(self) -> dict:
        """
        Get current queue statistics.

        Returns:
            Dictionary with queue size, in_progress count, and pending count
        """
        with self.lock:
            return {
                'queue_size': self.queue.qsize(),
                'in_progress': len(self.in_progress),
                'pending': len(self.pending),
                'total': self.queue.qsize() + len(self.in_progress)
            }

    def is_empty(self) -> bool:
        """Check if queue is empty and nothing is in progress."""
        with self.lock:
            return self.queue.empty() and len(self.in_progress) == 0

    def clear(self):
        """Clear all pending items from queue (for testing/shutdown)."""
        with self.lock:
            # Drain the queue
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

            self.pending.clear()
            # Note: Don't clear in_progress - let those finish naturally

        logger.info('Queue cleared')
