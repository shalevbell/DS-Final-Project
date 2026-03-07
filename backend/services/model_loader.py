"""
Model preloading service.

Preloads all ML models at application startup to eliminate first-chunk latency.
"""

import logging
import joblib
from pathlib import Path
from faster_whisper import WhisperModel
import mediapipe as mp

from config import Config

logger = logging.getLogger(__name__)


def preload_whisper_model() -> bool:
    """
    Preload Whisper model at startup.

    Sets analyze_audio_whisper._model attribute so lazy loading is skipped.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from run_models import analyze_audio_whisper

        logger.info(f'[ModelLoader] Preloading Whisper model: "{Config.WHISPER_MODEL_NAME}" '
                   f'({Config.WHISPER_DEVICE}, {Config.WHISPER_COMPUTE_TYPE})')

        model = WhisperModel(
            Config.WHISPER_MODEL_NAME,
            device=Config.WHISPER_DEVICE,
            compute_type=Config.WHISPER_COMPUTE_TYPE
        )

        # Set as function attribute (same pattern as lazy loading)
        analyze_audio_whisper._model = model

        logger.info('[ModelLoader] Whisper model preloaded successfully')
        return True

    except Exception as e:
        logger.error(f'[ModelLoader] Failed to preload Whisper model: {e}', exc_info=True)
        return False


def preload_mediapipe_models() -> bool:
    """
    Preload MediaPipe models at startup.

    Downloads models from Google Storage if needed and initializes landmarkers.
    Sets analyze_video_mediapipe._landmarkers attribute.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from run_models import analyze_video_mediapipe, _download_mediapipe_model

        logger.info('[ModelLoader] Preloading MediaPipe models...')

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
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=Config.MEDIAPIPE_FACE_MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=Config.MEDIAPIPE_FACE_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=Config.MEDIAPIPE_FACE_MIN_TRACKING_CONFIDENCE
        )

        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_pose_detection_confidence=Config.MEDIAPIPE_POSE_MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=Config.MEDIAPIPE_POSE_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=Config.MEDIAPIPE_POSE_MIN_TRACKING_CONFIDENCE
        )

        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=Config.MEDIAPIPE_HAND_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=Config.MEDIAPIPE_HAND_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=Config.MEDIAPIPE_HAND_MIN_TRACKING_CONFIDENCE
        )

        # Create landmarkers
        face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)
        pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
        hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

        # Set as function attribute (same pattern as lazy loading)
        analyze_video_mediapipe._landmarkers = {
            'face': face_landmarker,
            'pose': pose_landmarker,
            'hand': hand_landmarker
        }

        logger.info('[ModelLoader] MediaPipe models preloaded successfully')
        return True

    except Exception as e:
        logger.error(f'[ModelLoader] Failed to preload MediaPipe models: {e}', exc_info=True)
        return False


def preload_vocaltone_model() -> bool:
    """
    Preload VocalTone emotion classifier model at startup.

    Loads model, scaler, and labels from disk.
    Sets analyze_vocal_tone._model/_scaler/_labels_map attributes.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from run_models import analyze_vocal_tone

        logger.info('[ModelLoader] Preloading VocalTone model...')

        # Use VOCAL_TONE_MODEL_DIR if set, else default backend/vocal_tone_model/models
        if Config.VOCAL_TONE_MODEL_DIR:
            models_dir = Path(Config.VOCAL_TONE_MODEL_DIR)
        else:
            backend_dir = Path(__file__).parent.parent  # services -> backend
            models_dir = backend_dir / 'vocal_tone_model' / 'models'

        model_path = models_dir / 'vocal_tone_model.pkl'
        scaler_path = models_dir / 'vocal_tone_scaler.pkl'
        labels_path = models_dir / 'vocal_tone_labels.pkl'

        if not model_path.exists():
            logger.warning(f'[ModelLoader] Vocal Tone model not found: {model_path}. Skipping preload.')
            return False

        # Load model files
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        labels_map = joblib.load(labels_path)

        # Set as function attributes (same pattern as lazy loading)
        analyze_vocal_tone._model = model
        analyze_vocal_tone._scaler = scaler
        analyze_vocal_tone._labels_map = labels_map

        logger.info(f'[ModelLoader] VocalTone model preloaded: {type(model).__name__}, {len(labels_map)} classes')
        return True

    except Exception as e:
        logger.error(f'[ModelLoader] Failed to preload VocalTone model: {e}', exc_info=True)
        return False


def preload_all_models() -> dict:
    """
    Preload all ML models at application startup.

    This eliminates first-chunk latency by loading models before any chunks arrive.
    Models are set as function attributes (same pattern as lazy loading for compatibility).

    Errors during preloading are logged but don't crash app startup (graceful degradation).
    Lazy loading fallbacks remain as safety net.

    Returns:
        dict: Status of each model {'whisper': bool, 'mediapipe': bool, 'vocaltone': bool}
    """
    logger.info('=' * 56)
    logger.info('[ModelLoader] Starting model preloading...')
    logger.info('=' * 56)

    status = {
        'whisper': preload_whisper_model(),
        'mediapipe': preload_mediapipe_models(),
        'vocaltone': preload_vocaltone_model()
    }

    # Summary logging
    logger.info('=' * 56)
    successful = [name for name, success in status.items() if success]
    failed = [name for name, success in status.items() if not success]

    if successful:
        logger.info(f'[ModelLoader] Successfully preloaded: {", ".join(successful)}')
    if failed:
        logger.warning(f'[ModelLoader] Failed to preload: {", ".join(failed)}')

    logger.info('[ModelLoader] Model preloading complete')
    logger.info('=' * 56)

    return status
