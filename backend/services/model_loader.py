"""
Model preloading service.

Preloads all ML models at application startup to eliminate first-chunk latency.
"""

import logging
import joblib
from pathlib import Path
from faster_whisper import WhisperModel
import mediapipe as mp
import requests

from config import Config

logger = logging.getLogger(__name__)

# Global variable to track model preload status (accessible by API endpoints)
_preload_status = {
    'whisper': {'ready': False, 'loading': False, 'error': None},
    'mediapipe': {'ready': False, 'loading': False, 'error': None},
    'vocaltone': {'ready': False, 'loading': False, 'error': None},
    'ollama': {'ready': False, 'loading': False, 'error': None, 'progress': 0, 'status_text': 'Waiting...'}
}

def get_preload_status() -> dict:
    """Get current model preload status for API endpoints."""
    return _preload_status.copy()


def preload_whisper_model() -> bool:
    """
    Preload Whisper model at startup.

    Sets analyze_audio_whisper._model attribute so lazy loading is skipped.

    Returns:
        bool: True if successful, False otherwise
    """
    _preload_status['whisper']['loading'] = True
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
        _preload_status['whisper']['ready'] = True
        _preload_status['whisper']['loading'] = False
        return True

    except Exception as e:
        logger.error(f'[ModelLoader] Failed to preload Whisper model: {e}', exc_info=True)
        _preload_status['whisper']['error'] = str(e)
        _preload_status['whisper']['loading'] = False
        return False


def preload_mediapipe_models() -> bool:
    """
    Preload MediaPipe models at startup.

    Downloads models from Google Storage if needed and initializes landmarkers.
    Sets analyze_video_mediapipe._landmarkers attribute.

    Returns:
        bool: True if successful, False otherwise
    """
    _preload_status['mediapipe']['loading'] = True
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
        _preload_status['mediapipe']['ready'] = True
        _preload_status['mediapipe']['loading'] = False
        return True

    except Exception as e:
        logger.error(f'[ModelLoader] Failed to preload MediaPipe models: {e}', exc_info=True)
        _preload_status['mediapipe']['error'] = str(e)
        _preload_status['mediapipe']['loading'] = False
        return False


def preload_vocaltone_model() -> bool:
    """
    Preload VocalTone emotion classifier model at startup.

    Loads model, scaler, and labels from disk.
    Sets analyze_vocal_tone._model/_scaler/_labels_map attributes.

    Returns:
        bool: True if successful, False otherwise
    """
    _preload_status['vocaltone']['loading'] = True
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
            _preload_status['vocaltone']['error'] = 'Model files not found'
            _preload_status['vocaltone']['loading'] = False
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
        _preload_status['vocaltone']['ready'] = True
        _preload_status['vocaltone']['loading'] = False
        return True

    except Exception as e:
        logger.error(f'[ModelLoader] Failed to preload VocalTone model: {e}', exc_info=True)
        _preload_status['vocaltone']['error'] = str(e)
        _preload_status['vocaltone']['loading'] = False
        return False


def preload_ollama_model() -> bool:
    """
    Preload Ollama model by pulling it from Ollama registry.

    Ensures model is available before first chunk processing.
    Uses Ollama API to check if model exists, pulls if needed.

    Returns:
        bool: True if successful (model ready), False otherwise
    """
    _preload_status['ollama']['loading'] = True
    try:
        base_url = Config.OLLAMA_BASE_URL.rstrip("/")
        model_name = Config.OLLAMA_MODEL_NAME

        logger.info(f'[ModelLoader] Checking Ollama model: "{model_name}"')

        # Check if model exists (list local models)
        try:
            response = requests.get(
                f"{base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # Check if model is in local models list
            models = data.get("models", [])
            model_exists = any(
                model.get("name", "").startswith(model_name)
                for model in models
            )

            if model_exists:
                logger.info(f'[ModelLoader] Ollama model "{model_name}" already downloaded')
                _preload_status['ollama']['ready'] = True
                _preload_status['ollama']['loading'] = False
                return True

        except Exception as check_err:
            logger.warning(f'[ModelLoader] Could not check Ollama models: {check_err}')

        # Pull model if not found
        logger.info(f'[ModelLoader] Pulling Ollama model "{model_name}" (this may take several minutes)...')

        response = requests.post(
            f"{base_url}/api/pull",
            json={"name": model_name, "stream": True},
            timeout=600,  # 10 minutes timeout for model download
            stream=True
        )
        response.raise_for_status()

        # Process streaming response for progress updates
        import json
        total_size = 0
        downloaded_size = 0
        last_status = ""
        last_logged_pct = -1

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)

                    # Check for errors first
                    if "error" in data:
                        error_msg = data["error"]
                        logger.error(f'[ModelLoader] Ollama pull failed: {error_msg}')
                        _preload_status['ollama']['error'] = error_msg
                        _preload_status['ollama']['loading'] = False
                        _preload_status['ollama']['ready'] = False
                        return False

                    status = data.get("status", "")

                    # Update progress for download status
                    if "total" in data and "completed" in data:
                        total_size = data["total"]
                        downloaded_size = data["completed"]
                        progress_pct = int((downloaded_size / total_size * 100)) if total_size > 0 else 0

                        # Always update status (for frontend polling)
                        _preload_status['ollama']['progress'] = progress_pct
                        _preload_status['ollama']['status_text'] = status  # Don't include % here - frontend will show it

                        # Log progress every 10% (avoid spam)
                        if progress_pct // 10 != last_logged_pct // 10:
                            logger.info(f'[ModelLoader] Ollama pull progress: {progress_pct}% ({downloaded_size}/{total_size} bytes)')
                            last_logged_pct = progress_pct
                    elif status and status != last_status:
                        logger.info(f'[ModelLoader] Ollama: {status}')
                        _preload_status['ollama']['status_text'] = status
                        last_status = status

                except json.JSONDecodeError:
                    continue

        logger.info(f'[ModelLoader] Ollama model "{model_name}" pulled successfully')
        _preload_status['ollama']['ready'] = True
        _preload_status['ollama']['loading'] = False
        _preload_status['ollama']['progress'] = 100
        _preload_status['ollama']['status_text'] = 'Ready'
        return True

    except requests.exceptions.ConnectionError as e:
        logger.error(f'[ModelLoader] Cannot connect to Ollama service at {Config.OLLAMA_BASE_URL}: {e}')
        _preload_status['ollama']['error'] = f'Cannot connect to Ollama service: {e}'
        _preload_status['ollama']['loading'] = False
        return False
    except requests.exceptions.Timeout:
        logger.error('[ModelLoader] Ollama model pull timed out (>10 minutes)')
        _preload_status['ollama']['error'] = 'Model pull timed out'
        _preload_status['ollama']['loading'] = False
        return False
    except Exception as e:
        logger.error(f'[ModelLoader] Failed to preload Ollama model: {e}', exc_info=True)
        _preload_status['ollama']['error'] = str(e)
        _preload_status['ollama']['loading'] = False
        return False


def preload_all_models() -> dict:
    """
    Preload all ML models at application startup.

    This eliminates first-chunk latency by loading models before any chunks arrive.
    Models are set as function attributes (same pattern as lazy loading for compatibility).

    Errors during preloading are logged but don't crash app startup (graceful degradation).
    Lazy loading fallbacks remain as safety net.

    Returns:
        dict: Status of each model {'whisper': bool, 'mediapipe': bool, 'vocaltone': bool, 'ollama': bool}
    """
    logger.info('=' * 56)
    logger.info('[ModelLoader] Starting model preloading...')
    logger.info('=' * 56)

    status = {
        'whisper': preload_whisper_model(),
        'mediapipe': preload_mediapipe_models(),
        'vocaltone': preload_vocaltone_model(),
        'ollama': preload_ollama_model()
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
