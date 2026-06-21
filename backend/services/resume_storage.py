"""
Persist uploaded resume files per interview session.
"""

import logging
import os
import re
from typing import Optional, Tuple

from config import Config

logger = logging.getLogger(__name__)


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r'[^\w.\- ]', '_', (name or 'resume.pdf').strip())
    return cleaned or 'resume.pdf'


def save_session_resume(session_id: str, pdf_bytes: bytes, original_filename: str) -> Optional[str]:
    """
    Save resume PDF bytes for a session.

    Returns:
        Stored filename on success, None on failure.
    """
    if not session_id or not pdf_bytes:
        return None

    storage_dir = Config.RESUME_STORAGE_DIR
    try:
        os.makedirs(storage_dir, exist_ok=True)
        safe_name = _safe_filename(original_filename)
        stored_name = f'{session_id}_{safe_name}'
        path = os.path.join(storage_dir, stored_name)
        with open(path, 'wb') as f:
            f.write(pdf_bytes)
        logger.info(f'[ResumeStorage] Saved resume for {session_id}: {stored_name}')
        return stored_name
    except Exception as e:
        logger.warning(f'[ResumeStorage] Failed to save resume for {session_id}: {e}')
        return None


def get_resume_path(session_id: str, stored_filename: str) -> Optional[str]:
    """Return absolute path to a stored resume if it exists."""
    if not session_id or not stored_filename:
        return None
    if '..' in stored_filename or stored_filename.startswith('/'):
        return None
    path = os.path.join(Config.RESUME_STORAGE_DIR, stored_filename)
    return path if os.path.isfile(path) else None


def resolve_resume_for_session(session_id: str, stored_filename: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Resolve resume file path and download name for a session.

    Returns:
        Tuple of (absolute_path, download_name) or None.
    """
    if not stored_filename:
        return None
    path = get_resume_path(session_id, stored_filename)
    if not path:
        return None
    download_name = stored_filename.split('_', 1)[-1] if '_' in stored_filename else stored_filename
    return path, download_name
