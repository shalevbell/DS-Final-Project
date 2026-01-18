"""
Video processing module.

Handles chunk data parsing and FFmpeg conversion to MP4/WAV formats.
"""

import logging
import base64
import subprocess
from typing import Tuple

logger = logging.getLogger(__name__)


def parse_chunk_data(chunk_data) -> bytes:
    """
    Parse and normalize chunk data from various input formats.

    Handles different data formats sent by Socket.IO:
    - bytes: Direct binary data
    - list/bytearray: Array of bytes
    - dict: Socket.IO wrapped buffer with 'data' field
    - str: Base64-encoded string (legacy support)

    Args:
        chunk_data: Raw chunk data in any supported format

    Returns:
        Normalized bytes

    Raises:
        ValueError: If chunk data format is invalid or unsupported
    """
    if isinstance(chunk_data, bytes):
        return chunk_data

    elif isinstance(chunk_data, (list, bytearray)):
        return bytes(chunk_data)

    elif isinstance(chunk_data, dict):
        # Socket.IO might wrap ArrayBuffer in {'data': [...], 'type': 'Buffer'}
        if 'data' in chunk_data:
            return bytes(chunk_data['data'])
        else:
            raise ValueError(f'Unexpected dict format, keys: {list(chunk_data.keys())}')

    elif isinstance(chunk_data, str):
        # Legacy base64 support
        try:
            return base64.b64decode(chunk_data)
        except Exception as e:
            raise ValueError(f'Failed to decode base64 string: {e}')

    else:
        raise ValueError(f'Unsupported chunk data type: {type(chunk_data).__name__}')


def convert_chunk_with_ffmpeg(input_bytes: bytes, ffmpeg_path: str) -> Tuple[bytes, bytes]:
    """
    Convert video chunk to MP4 and extract audio to WAV using FFmpeg.

    Args:
        input_bytes: Raw video chunk data
        ffmpeg_path: Path to FFmpeg executable

    Returns:
        Tuple of (video_bytes, audio_bytes)
        - video_bytes: MP4-encoded video (fragmented for pipe compatibility)
        - audio_bytes: WAV-encoded audio

    Raises:
        subprocess.CalledProcessError: If FFmpeg conversion fails
    """
    # Convert video to MP4 using fragmented format for pipe compatibility
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

    # Extract audio to WAV
    audio_proc = subprocess.run(
        [
            ffmpeg_path, '-hide_banner', '-loglevel', 'error',
            '-i', 'pipe:0',
            '-vn',  # No video
            '-map', '0:a:0',  # Map first audio stream
            '-c:a', 'pcm_s16le',  # WAV codec (PCM 16-bit little-endian)
            '-ar', '44100',  # Standard sample rate
            '-ac', '2',  # Stereo (2 channels)
            '-f', 'wav', 'pipe:1'
        ],
        input=input_bytes,
        capture_output=True,
        check=True
    )
    audio_bytes = audio_proc.stdout

    logger.debug(f'FFmpeg conversion complete: {len(video_bytes)} video bytes, {len(audio_bytes)} audio bytes')

    return video_bytes, audio_bytes
