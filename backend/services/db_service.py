"""
Database service module.

Centralized PostgreSQL operations for session and chunk result persistence.
All functions degrade gracefully on DB failure so the app continues running.
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

_database_url: Optional[str] = None

DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id              SERIAL PRIMARY KEY,
    session_id      VARCHAR(64) NOT NULL UNIQUE,
    candidate_name  VARCHAR(255) NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    status          VARCHAR(16) NOT NULL DEFAULT 'active',
    video_enabled   BOOLEAN DEFAULT TRUE,
    audio_enabled   BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_sessions_candidate_name
    ON sessions(candidate_name);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at
    ON sessions(started_at DESC);

CREATE TABLE IF NOT EXISTS chunk_results (
    id                SERIAL PRIMARY KEY,
    session_id        VARCHAR(64) NOT NULL
                          REFERENCES sessions(session_id) ON DELETE CASCADE,
    chunk_index       INTEGER NOT NULL,
    processed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_results     JSONB NOT NULL DEFAULT '{}',
    processing_status VARCHAR(16) NOT NULL DEFAULT 'completed',
    UNIQUE(session_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunk_results_session_id
    ON chunk_results(session_id);
"""


def init_db(database_url: str) -> None:
    """
    Initialize the database: store the URL and create tables.

    Args:
        database_url: PostgreSQL connection URL
    """
    global _database_url
    _database_url = database_url
    _create_tables()
    logger.info('[DB] Database initialized successfully')


@contextmanager
def _get_connection():
    """Context manager that yields a psycopg2 connection and handles commit/rollback."""
    if not _database_url:
        raise RuntimeError('DB not initialized — call init_db() first')
    conn = psycopg2.connect(_database_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _create_tables() -> None:
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)


def save_session(
    session_id: str,
    candidate_name: str,
    started_at: datetime,
    video_enabled: bool = True,
    audio_enabled: bool = True,
) -> bool:
    """
    Insert a new session record. Silently ignores duplicate session_id.

    Returns:
        True on success, False on error
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions
                        (session_id, candidate_name, started_at, video_enabled, audio_enabled)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    (session_id, candidate_name, started_at, video_enabled, audio_enabled),
                )
        return True
    except Exception as e:
        logger.warning(f'[DB] save_session failed for {session_id}: {e}')
        return False


def complete_session(session_id: str) -> bool:
    """
    Mark a session as completed with the current timestamp.

    Returns:
        True on success, False on error
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE sessions
                       SET ended_at = NOW(), status = 'completed'
                     WHERE session_id = %s
                    """,
                    (session_id,),
                )
        return True
    except Exception as e:
        logger.warning(f'[DB] complete_session failed for {session_id}: {e}')
        return False


def save_chunk_results(
    session_id: str,
    chunk_index: int,
    model_results: Dict,
    status: str = 'completed',
) -> bool:
    """
    Upsert chunk results. Safe to call multiple times (retries).

    Args:
        session_id: Session identifier
        chunk_index: Chunk index within the session
        model_results: Dict keyed by model name containing all model outputs
        status: 'completed' or 'failed'

    Returns:
        True on success, False on error
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chunk_results
                        (session_id, chunk_index, model_results, processing_status)
                    VALUES (%s, %s, %s::jsonb, %s)
                    ON CONFLICT (session_id, chunk_index) DO UPDATE
                        SET model_results     = EXCLUDED.model_results,
                            processed_at      = NOW(),
                            processing_status = EXCLUDED.processing_status
                    """,
                    (session_id, chunk_index, json.dumps(model_results), status),
                )
        return True
    except Exception as e:
        logger.warning(f'[DB] save_chunk_results failed for {session_id}:{chunk_index}: {e}')
        return False


def list_sessions(
    candidate_filter: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Optional[Tuple[List[Dict], int]]:
    """
    List sessions with optional case-insensitive candidate name filter.

    Returns:
        Tuple of (sessions list, total count) or None on error.
        Each session dict includes a 'chunk_count' key.
    """
    try:
        where_clause = ''
        params: list = []
        count_params: list = []

        if candidate_filter:
            where_clause = 'WHERE s.candidate_name ILIKE %s'
            params.append(f'%{candidate_filter}%')
            count_params.append(f'%{candidate_filter}%')

        with _get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT s.session_id, s.candidate_name, s.started_at, s.ended_at,
                           s.status, s.video_enabled, s.audio_enabled,
                           COUNT(c.id) AS chunk_count
                      FROM sessions s
                      LEFT JOIN chunk_results c ON s.session_id = c.session_id
                    {where_clause}
                     GROUP BY s.id
                     ORDER BY s.started_at DESC
                     LIMIT %s OFFSET %s
                    """,
                    params + [limit, offset],
                )
                rows = [dict(r) for r in cur.fetchall()]

                cur.execute(
                    f'SELECT COUNT(*) FROM sessions s {where_clause}',
                    count_params,
                )
                count_row = cur.fetchone()
                total = count_row['count'] if count_row else 0

        return rows, int(total)
    except Exception as e:
        logger.warning(f'[DB] list_sessions failed: {e}')
        return None


def get_session_with_chunks(session_id: str) -> Optional[Dict]:
    """
    Retrieve a session and all its chunk results.

    Returns:
        Dict with 'session' and 'chunks' keys, or None on error/not found.
    """
    try:
        with _get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    'SELECT * FROM sessions WHERE session_id = %s',
                    (session_id,),
                )
                session_row = cur.fetchone()
                if not session_row:
                    return None

                cur.execute(
                    """
                    SELECT chunk_index, processed_at, processing_status, model_results
                      FROM chunk_results
                     WHERE session_id = %s
                     ORDER BY chunk_index ASC
                    """,
                    (session_id,),
                )
                chunk_rows = [dict(r) for r in cur.fetchall()]

        return {'session': dict(session_row), 'chunks': chunk_rows}
    except Exception as e:
        logger.warning(f'[DB] get_session_with_chunks failed for {session_id}: {e}')
        return None


def get_chunk_detail(session_id: str, chunk_index: int) -> Optional[Dict]:
    """
    Retrieve a single chunk result.

    Returns:
        Chunk dict or None on error/not found.
    """
    try:
        with _get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT chunk_index, processed_at, processing_status, model_results
                      FROM chunk_results
                     WHERE session_id = %s AND chunk_index = %s
                    """,
                    (session_id, chunk_index),
                )
                row = cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        logger.warning(f'[DB] get_chunk_detail failed for {session_id}:{chunk_index}: {e}')
        return None


def delete_session(session_id: str) -> bool:
    """
    Delete a session and all its chunk results (cascades via FK).

    Returns:
        True on success, False on error
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM sessions WHERE session_id = %s', (session_id,))
        return True
    except Exception as e:
        logger.warning(f'[DB] delete_session failed for {session_id}: {e}')
        return False


def rename_session(session_id: str, candidate_name: str) -> bool:
    """
    Update the candidate name for a session.

    Returns:
        True on success, False on error
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'UPDATE sessions SET candidate_name = %s WHERE session_id = %s',
                    (candidate_name, session_id),
                )
        return True
    except Exception as e:
        logger.warning(f'[DB] rename_session failed for {session_id}: {e}')
        return False
