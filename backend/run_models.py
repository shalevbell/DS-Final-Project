"""
Analysis functions for chunk processing.

Provides interfaces for ML models and functions neeeded for chunk processing.
"""

import logging
import os
import tempfile
import time
import traceback
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
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


def _download_mediapipe_model(model_name: str, url: str, models_dir: Path) -> bool:
    """
    Download MediaPipe model if not present.

    Args:
        model_name: Name of the model file
        url: Download URL
        models_dir: Directory to store model

    Returns:
        True if downloaded successfully or already exists, False otherwise
    """
    model_path = models_dir / model_name

    if model_path.exists():
        return True

    try:
        logger.info(f'[MediaPipe] Downloading {model_name}...')
        urllib.request.urlretrieve(url, model_path)
        logger.info(f'[MediaPipe] Downloaded {model_name}')
        return True
    except Exception as e:
        logger.error(f'[MediaPipe] Error downloading {model_name}: {e}')
        return False


def _analyze_face_simple(face_landmarks_list: List) -> Dict:
    """
    Simplified facial analysis for emotion detection.

    Args:
        face_landmarks_list: List of detected face landmarks

    Returns:
        Dictionary with emotion and confidence
    """
    if not face_landmarks_list or len(face_landmarks_list) == 0:
        return {'emotion': 'neutral', 'confidence': 0.0, 'detected': False}

    face_landmarks = face_landmarks_list[0]
    emotions = []

    try:
        # Mouth analysis for smile/frown
        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]
        left_mouth_corner = face_landmarks[61]
        right_mouth_corner = face_landmarks[291]

        mouth_height = abs(upper_lip.y - lower_lip.y)
        mouth_center_y = (upper_lip.y + lower_lip.y) / 2
        corners_avg_y = (left_mouth_corner.y + right_mouth_corner.y) / 2
        smile_ratio = mouth_center_y - corners_avg_y

        if smile_ratio > 0.01:
            emotions.append('happy')
        elif smile_ratio < -0.01:
            emotions.append('sad')

        if mouth_height > 0.04:
            emotions.append('surprised')

        # Eyebrow analysis
        left_eyebrow = face_landmarks[70]
        right_eyebrow = face_landmarks[300]
        left_eye_top = face_landmarks[159]
        right_eye_top = face_landmarks[386]

        avg_brow_distance = (abs(left_eyebrow.y - left_eye_top.y) +
                            abs(right_eyebrow.y - right_eye_top.y)) / 2

        if avg_brow_distance > 0.06:
            emotions.append('surprised')

        # Default to neutral if no strong emotion
        if not emotions:
            emotions.append('neutral')
        else:
            emotions.append('engaged')

        # Get most common emotion
        emotion_counts = Counter(emotions)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        confidence = min(emotion_counts[dominant_emotion] / len(emotions), 1.0)

        return {
            'emotion': dominant_emotion,
            'confidence': confidence,
            'detected': True,
            'all_emotions': list(set(emotions))
        }

    except Exception as e:
        logger.debug(f'[MediaPipe] Face analysis error: {e}')
        return {'emotion': 'neutral', 'confidence': 0.0, 'detected': True}


def _analyze_posture_simple(pose_landmarks_list: List) -> Dict:
    """
    Simplified posture analysis.

    Args:
        pose_landmarks_list: List of detected pose landmarks

    Returns:
        Dictionary with posture score
    """
    if not pose_landmarks_list or len(pose_landmarks_list) == 0:
        return {'score': 0.5, 'detected': False}

    pose_landmarks = pose_landmarks_list[0]
    score = 0.5  # Default fair posture

    try:
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        left_hip = pose_landmarks[23]
        right_hip = pose_landmarks[24]

        # Check body lean
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        lean = abs(shoulder_center_x - hip_center_x)

        if lean < 0.03:
            score = 0.9  # Good upright posture
        elif lean < 0.08:
            score = 0.6  # Fair posture
        else:
            score = 0.3  # Poor posture

        return {'score': score, 'detected': True}

    except Exception as e:
        logger.debug(f'[MediaPipe] Posture analysis error: {e}')
        return {'score': 0.5, 'detected': True}


def _analyze_hands_simple(hand_landmarks_list: List) -> Dict:
    """
    Simplified hand gesture analysis.

    Args:
        hand_landmarks_list: List of detected hand landmarks

    Returns:
        Dictionary with hand gesture info
    """
    if not hand_landmarks_list or len(hand_landmarks_list) == 0:
        return {'detected': False, 'count': 0}

    return {
        'detected': True,
        'count': len(hand_landmarks_list)
    }


def analyze_video_mediapipe(video_bytes: bytes, session_id: str, chunk_index: int) -> Dict:
    """
    Analyze video for facial expressions, gestures, and posture using MediaPipe.

    Args:
        video_bytes: MP4 video data
        session_id: Session identifier
        chunk_index: Chunk index within session

    Returns:
        Dictionary with video analysis results:
        {
            'emotions': list,
            'dominant_emotion': str,
            'emotion_confidence': float,
            'posture_score': float,
            'engagement_score': float,
            'hand_gestures_detected': bool,
            'facial_landmarks_detected': bool,
            'pose_landmarks_detected': bool,
            'frames_analyzed': int,
            'processing_time_ms': int
        }
    """
    start_time = time.time()
    temp_path = None

    try:
        logger.info(f'[MediaPipe] Processing chunk {session_id}:{chunk_index} ({len(video_bytes)} bytes)')

        # Initialize MediaPipe models on first use (lazy loading)
        if not hasattr(analyze_video_mediapipe, '_landmarkers'):
            logger.info('[MediaPipe] Initializing models (first use)...')

            # Create models directory
            models_dir = Path(Config.MEDIAPIPE_MODEL_DIR)
            models_dir.mkdir(exist_ok=True, parents=True)

            # Model URLs
            model_urls = {
                Config.MEDIAPIPE_FACE_MODEL: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                Config.MEDIAPIPE_POSE_MODEL: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
                Config.MEDIAPIPE_HAND_MODEL: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            }

            # Download models if needed
            for model_name, url in model_urls.items():
                if not _download_mediapipe_model(model_name, url, models_dir):
                    raise RuntimeError(f'Failed to download {model_name}')

            # Initialize landmarkers
            base_options_face = mp.tasks.BaseOptions(
                model_asset_path=str(models_dir / Config.MEDIAPIPE_FACE_MODEL)
            )
            base_options_pose = mp.tasks.BaseOptions(
                model_asset_path=str(models_dir / Config.MEDIAPIPE_POSE_MODEL)
            )
            base_options_hand = mp.tasks.BaseOptions(
                model_asset_path=str(models_dir / Config.MEDIAPIPE_HAND_MODEL)
            )

            face_options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options_face,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=Config.MEDIAPIPE_FACE_MIN_DETECTION_CONFIDENCE,
                min_face_presence_confidence=Config.MEDIAPIPE_FACE_MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=Config.MEDIAPIPE_FACE_MIN_TRACKING_CONFIDENCE
            )

            pose_options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=base_options_pose,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                min_pose_detection_confidence=Config.MEDIAPIPE_POSE_MIN_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=Config.MEDIAPIPE_POSE_MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=Config.MEDIAPIPE_POSE_MIN_TRACKING_CONFIDENCE
            )

            hand_options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_options_hand,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=Config.MEDIAPIPE_HAND_MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=Config.MEDIAPIPE_HAND_MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=Config.MEDIAPIPE_HAND_MIN_TRACKING_CONFIDENCE
            )

            analyze_video_mediapipe._landmarkers = {
                'face': mp.tasks.vision.FaceLandmarker.create_from_options(face_options),
                'pose': mp.tasks.vision.PoseLandmarker.create_from_options(pose_options),
                'hand': mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
            }

            logger.info('[MediaPipe] Models initialized successfully')

        # Save video bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            temp_path = tmp_file.name

        # Open video with OpenCV
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError(f'Could not open video file')

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0

        logger.info(
            f'[MediaPipe] Video properties for {session_id}:{chunk_index}: '
            f'{total_frames} frames, {fps:.2f} fps, {duration_sec:.2f}s duration, '
            f'sample_rate={Config.MEDIAPIPE_FRAME_SAMPLE_RATE}'
        )

        frame_count = 0
        frames_analyzed = 0

        # Aggregate results
        all_face_results = []
        all_pose_results = []
        all_hand_results = []

        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample every Nth frame
            if frame_count % Config.MEDIAPIPE_FRAME_SAMPLE_RATE == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Calculate timestamp in milliseconds
                timestamp_ms = int((frame_count / fps) * 1000) if fps > 0 else frame_count * 33

                # Process with landmarkers
                try:
                    face_result = analyze_video_mediapipe._landmarkers['face'].detect_for_video(mp_image, timestamp_ms)
                    pose_result = analyze_video_mediapipe._landmarkers['pose'].detect_for_video(mp_image, timestamp_ms)
                    hand_result = analyze_video_mediapipe._landmarkers['hand'].detect_for_video(mp_image, timestamp_ms)

                    all_face_results.append(face_result.face_landmarks)
                    all_pose_results.append(pose_result.pose_landmarks)
                    all_hand_results.append(hand_result.hand_landmarks)

                    frames_analyzed += 1
                except Exception as e:
                    logger.debug(f'[MediaPipe] Frame {frame_count} processing error: {e}')

            frame_count += 1

        cap.release()

        # Analyze aggregated results
        emotions_list = []
        emotion_confidences = []
        posture_scores = []
        hand_detected_count = 0

        for face_landmarks in all_face_results:
            face_analysis = _analyze_face_simple(face_landmarks)
            if face_analysis['detected']:
                emotions_list.append(face_analysis['emotion'])
                emotion_confidences.append(face_analysis['confidence'])

        for pose_landmarks in all_pose_results:
            posture_analysis = _analyze_posture_simple(pose_landmarks)
            if posture_analysis['detected']:
                posture_scores.append(posture_analysis['score'])

        for hand_landmarks in all_hand_results:
            hand_analysis = _analyze_hands_simple(hand_landmarks)
            if hand_analysis['detected']:
                hand_detected_count += 1

        # Calculate final metrics
        dominant_emotion = Counter(emotions_list).most_common(1)[0][0] if emotions_list else 'neutral'
        avg_emotion_confidence = sum(emotion_confidences) / len(emotion_confidences) if emotion_confidences else 0.0
        avg_posture_score = sum(posture_scores) / len(posture_scores) if posture_scores else 0.5
        hand_gestures_detected = hand_detected_count > 0

        # Log analysis results
        logger.info(
            f'[MediaPipe] Analysis: emotion={dominant_emotion} (conf={avg_emotion_confidence:.2f}), '
            f'posture={avg_posture_score:.2f}, hands={hand_gestures_detected}, '
            f'face_detected={len(emotions_list) > 0}, pose_detected={len(posture_scores) > 0}'
        )

        # Calculate engagement score
        face_visibility_score = len(emotions_list) / max(frames_analyzed, 1)
        positive_emotion_score = 1.0 if dominant_emotion in ['happy', 'engaged'] else 0.5
        posture_contribution = avg_posture_score
        hand_contribution = 1.0 if hand_gestures_detected else 0.5

        engagement_score = (
            face_visibility_score * 0.2 +
            positive_emotion_score * 0.3 +
            posture_contribution * 0.3 +
            hand_contribution * 0.2
        )

        processing_time = int((time.time() - start_time) * 1000)

        result = {
            'emotions': list(set(emotions_list)) if emotions_list else ['neutral'],
            'dominant_emotion': dominant_emotion,
            'emotion_confidence': round(avg_emotion_confidence, 2),
            'posture_score': round(avg_posture_score, 2),
            'engagement_score': round(engagement_score, 2),
            'hand_gestures_detected': hand_gestures_detected,
            'facial_landmarks_detected': len(emotions_list) > 0,
            'pose_landmarks_detected': len(posture_scores) > 0,
            'frames_analyzed': frames_analyzed,
            'processing_time_ms': processing_time
        }

        logger.info(
            f'[MediaPipe] Completed chunk {session_id}:{chunk_index} in {processing_time}ms '
            f'(analyzed {frames_analyzed} frames, emotion: {dominant_emotion}, '
            f'posture: {avg_posture_score:.2f}, engagement: {engagement_score:.2f})'
        )

        return result

    except Exception as e:
        logger.error(
            f'[MediaPipe] Error processing chunk {session_id}:{chunk_index}: {e}\n'
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
                logger.debug(f'[MediaPipe] Removed temp file: {temp_path}')
            except Exception as cleanup_error:
                logger.warning(f'[MediaPipe] Temp cleanup failed: {cleanup_error}')


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
