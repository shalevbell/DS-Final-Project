"""
Analysis functions for chunk processing.

Provides interfaces for ML models and functions neeeded for chunk processing.
"""

import logging
import os
import tempfile
import time
import traceback
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

from faster_whisper import WhisperModel

from config import Config

logger = logging.getLogger(__name__)


def get_available_models():
    """
    Get list of available analysis models.

    Returns:
        List of model names that can be used for chunk analysis
    """
    # MODEL_REGISTRY is defined after functions but will be available at runtime
    return list(MODEL_REGISTRY.keys())


def analyze_audio_whisper(audio_bytes: bytes, session_id: str, chunk_index: int) -> Dict:
    """
    Transcribe audio using Whisper.

    TODO: Integrate actual Whisper model from /test/whisper_test.py

    Args:
        audio_bytes: MP3 audio data
        session_id: Session identifier
        chunk_index: Chunk index within session

    Returns:
        Dictionary with transcription results:
        {
            'transcript': str,
            'confidence': float,
            'language': str,
            'segments': list,
            'processing_time_ms': int
        }
    """
    start_time = time.time()
    temp_path = None

    try:
        logger.info(f'[Whisper] Start chunk {session_id}:{chunk_index}')

        if isinstance(audio_bytes, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                temp_path = tmp_file.name
            logger.info(
                f'[Whisper] Saved audio to temp file: {temp_path} '
                f'({len(audio_bytes)} bytes)'
            )
            audio_source = temp_path
        elif isinstance(audio_bytes, str):
            audio_source = audio_bytes
            logger.info(f'[Whisper] Using audio path: {audio_source}')
        else:
            raise ValueError(f'Unsupported audio input type: {type(audio_bytes).__name__}')

        if not hasattr(analyze_audio_whisper, '_model'):
            logger.info(
                f'[Whisper] Loading faster-whisper model "{Config.WHISPER_MODEL_NAME}" ({Config.WHISPER_DEVICE}). '
                'First run can be slow due to model download.'
            )
            analyze_audio_whisper._model = WhisperModel(
                Config.WHISPER_MODEL_NAME,
                device=Config.WHISPER_DEVICE,
                compute_type=Config.WHISPER_COMPUTE_TYPE
            )

        model = analyze_audio_whisper._model

        logger.info(f'[Whisper] Starting transcription: {session_id}:{chunk_index}')

        segments, info = model.transcribe(
            audio_source,
            beam_size=1,
            vad_filter=True,
            word_timestamps=False
        )

        segment_list = []
        transcript_parts = []
        for segment in segments:
            segment_list.append(
                {
                    'start': float(segment.start),
                    'end': float(segment.end),
                    'text': segment.text
                }
            )
            transcript_parts.append(segment.text)

        transcript_text = ' '.join(part.strip() for part in transcript_parts if part).strip()
        if transcript_text:
            logger.info(f'[Whisper] Transcript: {transcript_text}')

        processing_time_sec = time.time() - start_time
        audio_duration = getattr(info, 'duration', None)
        audio_duration_sec = float(audio_duration) if audio_duration else None
        rtf = (processing_time_sec / audio_duration_sec) if audio_duration_sec else None
        audio_duration_label = f'{audio_duration_sec:.2f}s' if audio_duration_sec else 'n/a'
        rtf_label = f'{rtf:.3f}' if rtf is not None else 'n/a'

        logger.info(
            f'[Whisper] Completed chunk {session_id}:{chunk_index} '
            f'in {processing_time_sec:.2f}s, '
            f'audio_duration={audio_duration_label}, '
            f'RTF={rtf_label}'
        )

        result = {
            'transcript': transcript_text,
            'confidence': getattr(info, 'language_probability', None),
            'language': getattr(info, 'language', 'en'),
            'segments': segment_list,
            'processing_time_ms': int(processing_time_sec * 1000)
        }

        return result

    except Exception as e:
        logger.error(
            f'[Whisper] Error processing chunk {session_id}:{chunk_index}: {e}\n'
            f'{traceback.format_exc()}'
        )
        return {
            'error': str(e),
            'processing_time_ms': int((time.time() - start_time) * 1000)
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f'[Whisper] Removed temp file: {temp_path}')
            except Exception as cleanup_error:
                logger.warning(f'[Whisper] Temp cleanup failed: {cleanup_error}')


def analyze_video_mediapipe(video_bytes: bytes, session_id: str, chunk_index: int) -> Dict:
    """
    Analyze video for facial expressions, gestures, and posture using MediaPipe.

    TODO: Integrate actual MediaPipe model from /test/video_analysis.py

    Args:
        video_bytes: MP4 video data
        session_id: Session identifier
        chunk_index: Chunk index within session

    Returns:
        Dictionary with video analysis results:
        {
            'emotions': list,
            'gestures': list,
            'posture_score': float,
            'eye_contact_score': float,
            'engagement_score': float,
            'processing_time_ms': int
        }
    """
    start_time = time.time()

    try:
        logger.info(f'[MediaPipe] Processing chunk {session_id}:{chunk_index} ({len(video_bytes)} bytes)')

        # TODO: Replace with actual MediaPipe implementation
        # Example integration:
        # import mediapipe as mp
        # mp_holistic = mp.solutions.holistic
        # with mp_holistic.Holistic() as holistic:
        #     results = holistic.process(frame)

        # Placeholder implementation
        time.sleep(0.15)  # Simulate processing time

        processing_time = int((time.time() - start_time) * 1000)

        result = {
            'emotions': ['neutral', 'engaged'],
            'gestures': ['talking', 'hand_gesture'],
            'posture_score': 0.8,
            'eye_contact_score': 0.7,
            'engagement_score': 0.75,
            'facial_landmarks_detected': True,
            'pose_landmarks_detected': True,
            'processing_time_ms': processing_time
        }

        logger.info(f'[MediaPipe] Completed chunk {session_id}:{chunk_index} in {processing_time}ms')
        return result

    except Exception as e:
        logger.error(f'[MediaPipe] Error processing chunk {session_id}:{chunk_index}: {e}')
        return {
            'error': str(e),
            'processing_time_ms': int((time.time() - start_time) * 1000)
        }


def analyze_vocal_tone(audio_bytes: bytes, session_id: str, chunk_index: int) -> Dict:
    """
    Analyze vocal characteristics: pitch, tempo, energy, and emotional tone.

    TODO: Integrate actual vocal tone analysis model

    Args:
        audio_bytes: MP3 audio data
        session_id: Session identifier
        chunk_index: Chunk index within session

    Returns:
        Dictionary with vocal analysis results:
        {
            'pitch_mean': float,
            'pitch_std': float,
            'tempo': float,
            'energy_level': float,
            'valence': float,
            'arousal': float,
            'confidence_score': float,
            'processing_time_ms': int
        }
    """
    start_time = time.time()

    try:
        logger.info(f'[VocalTone] Processing chunk {session_id}:{chunk_index} ({len(audio_bytes)} bytes)')

        # TODO: Replace with actual vocal tone analysis
        # Example integration:
        # import librosa
        # y, sr = librosa.load(audio_file)
        # pitch = librosa.piptrack(y=y, sr=sr)
        # tempo = librosa.beat.tempo(y=y, sr=sr)

        # Placeholder implementation
        time.sleep(0.08)  # Simulate processing time

        processing_time = int((time.time() - start_time) * 1000)

        result = {
            'pitch_mean': 220.0,  # Hz
            'pitch_std': 25.0,
            'tempo': 120.0,  # BPM
            'energy_level': 0.6,  # 0-1
            'valence': 0.65,  # Emotional positivity (0-1)
            'arousal': 0.58,  # Emotional intensity (0-1)
            'confidence_score': 0.85,
            'processing_time_ms': processing_time
        }

        logger.info(f'[VocalTone] Completed chunk {session_id}:{chunk_index} in {processing_time}ms')
        return result

    except Exception as e:
        logger.error(f'[VocalTone] Error processing chunk {session_id}:{chunk_index}: {e}')
        return {
            'error': str(e),
            'processing_time_ms': int((time.time() - start_time) * 1000)
        }


# =============================================================================
# MODEL REGISTRY - Populate after functions are defined
# =============================================================================
#
# To add a new model:
# 1. Implement the analysis function above (signature: func(bytes, session_id, chunk_index) -> Dict)
# 2. Add an entry here mapping model name to:
#    - 'function': The analysis function to call
#    - 'data_type': Either 'audio' or 'video' - specifies which bytes to pass to the function
#
MODEL_REGISTRY = {
    'whisper': {
        'function': analyze_audio_whisper,
        'data_type': 'audio'
    },
    'mediapipe': {
        'function': analyze_video_mediapipe,
        'data_type': 'video'
    },
    'vocaltone': {
        'function': analyze_vocal_tone,
        'data_type': 'audio'
    }
}


def run_parallel_analysis(
    video_bytes: bytes,
    audio_bytes: bytes,
    session_id: str,
    chunk_index: int,
    max_workers: int = None,
    timeout: int = 60
) -> Dict:
    """
    Run all analysis functions in parallel using ThreadPoolExecutor.

    This provides function-level parallelism within a single chunk.

    Args:
        video_bytes: MP4 video data
        audio_bytes: MP3 audio data
        session_id: Session identifier
        chunk_index: Chunk index within session
        max_workers: Maximum parallel workers (default: None = auto-scale to model count)
        timeout: Timeout in seconds per function (default: 60)

    Returns:
        Dictionary with all analysis results:
        {
            'whisper': dict,
            'mediapipe': dict,
            'vocaltone': dict,
            'total_processing_time_ms': int
        }
    """
    overall_start = time.time()

    logger.info(f'Running parallel analysis for chunk {session_id}:{chunk_index}')

    results = {}

    # Auto-scale max_workers to match number of models
    if max_workers is None:
        max_workers = len(MODEL_REGISTRY)
        logger.debug(f'Auto-scaled max_workers to {max_workers} (matches model count)')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Dynamically build futures from MODEL_REGISTRY
        futures = {}
        for model_name, model_config in MODEL_REGISTRY.items():
            func = model_config['function']
            data_type = model_config['data_type']

            # Select the correct data bytes based on model's data_type
            data_bytes = audio_bytes if data_type == 'audio' else video_bytes

            futures[model_name] = executor.submit(func, data_bytes, session_id, chunk_index)

        # Collect results as they complete
        for name, future in futures.items():
            try:
                result = future.result(timeout=timeout)
                results[name] = result

                # Log success or error
                if 'error' in result:
                    logger.warning(f'[{name}] Failed for chunk {session_id}:{chunk_index}: {result["error"]}')
                else:
                    logger.debug(f'[{name}] Succeeded for chunk {session_id}:{chunk_index}')

            except TimeoutError:
                error_msg = f'Timeout after {timeout}s'
                logger.error(f'[{name}] Timeout for chunk {session_id}:{chunk_index}')
                results[name] = {'error': error_msg}

            except Exception as e:
                logger.error(f'[{name}] Exception for chunk {session_id}:{chunk_index}: {e}')
                results[name] = {'error': str(e)}

    total_time = int((time.time() - overall_start) * 1000)
    results['total_processing_time_ms'] = total_time

    # Calculate success count
    success_count = sum(1 for r in results.values() if isinstance(r, dict) and 'error' not in r)
    total_models = len(MODEL_REGISTRY)
    logger.info(
        f'Parallel analysis complete for chunk {session_id}:{chunk_index}: '
        f'{success_count}/{total_models} succeeded in {total_time}ms'
    )

    return results
