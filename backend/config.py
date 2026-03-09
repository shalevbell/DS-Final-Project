import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration with sensible defaults."""

    # Database configuration
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/interview_db"
    )

    # Redis configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = FLASK_ENV == "development"

    # CORS configuration
    CORS_ORIGINS = "*"

    # Chunk processing configuration
    PROCESSING_MAX_WORKERS = int(os.getenv("PROCESSING_MAX_WORKERS", "3"))
    PROCESSING_QUEUE_SIZE = int(os.getenv("PROCESSING_QUEUE_SIZE", "100"))
    PROCESSING_RETRY_ATTEMPTS = int(os.getenv("PROCESSING_RETRY_ATTEMPTS", "3"))
    PROCESSING_RETRY_DELAY = int(os.getenv("PROCESSING_RETRY_DELAY", "5"))  # seconds

    # PUBSUB configuration
    PUBSUB_CHANNEL = os.getenv("PUBSUB_CHANNEL", "chunks:ready")
    PUBSUB_RECONNECT_DELAY = int(os.getenv("PUBSUB_RECONNECT_DELAY", "5"))  # seconds

    # Chunk configuration
    CHUNK_DURATION_MS = 30000  # Video chunk duration in milliseconds (30 seconds)

    # Whisper model configuration
    WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base.en")
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # MediaPipe model configuration
    MEDIAPIPE_MODEL_DIR = os.getenv("MEDIAPIPE_MODEL_DIR", "/app/mediapipe_models")
    MEDIAPIPE_FACE_MODEL = os.getenv("MEDIAPIPE_FACE_MODEL", "face_landmarker.task")
    MEDIAPIPE_POSE_MODEL = os.getenv(
        "MEDIAPIPE_POSE_MODEL", "pose_landmarker_lite.task"
    )
    MEDIAPIPE_HAND_MODEL = os.getenv("MEDIAPIPE_HAND_MODEL", "hand_landmarker.task")
    MEDIAPIPE_FRAME_SAMPLE_RATE = int(os.getenv("MEDIAPIPE_FRAME_SAMPLE_RATE", "2"))

    # MediaPipe detection thresholdsj
    MEDIAPIPE_FACE_MIN_DETECTION_CONFIDENCE = float(
        os.getenv("MEDIAPIPE_FACE_MIN_DETECTION_CONFIDENCE", "0.5")
    )
    MEDIAPIPE_FACE_MIN_PRESENCE_CONFIDENCE = float(
        os.getenv("MEDIAPIPE_FACE_MIN_PRESENCE_CONFIDENCE", "0.5")
    )
    MEDIAPIPE_FACE_MIN_TRACKING_CONFIDENCE = float(
        os.getenv("MEDIAPIPE_FACE_MIN_TRACKING_CONFIDENCE", "0.5")
    )
    MEDIAPIPE_POSE_MIN_DETECTION_CONFIDENCE = float(
        os.getenv("MEDIAPIPE_POSE_MIN_DETECTION_CONFIDENCE", "0.5")
    )
    MEDIAPIPE_POSE_MIN_PRESENCE_CONFIDENCE = float(
        os.getenv("MEDIAPIPE_POSE_MIN_PRESENCE_CONFIDENCE", "0.5")
    )
    MEDIAPIPE_POSE_MIN_TRACKING_CONFIDENCE = float(
        os.getenv("MEDIAPIPE_POSE_MIN_TRACKING_CONFIDENCE", "0.5")
    )
    MEDIAPIPE_HAND_MIN_DETECTION_CONFIDENCE = float(
        os.getenv("MEDIAPIPE_HAND_MIN_DETECTION_CONFIDENCE", "0.5")
    )
    MEDIAPIPE_HAND_MIN_PRESENCE_CONFIDENCE = float(
        os.getenv("MEDIAPIPE_HAND_MIN_PRESENCE_CONFIDENCE", "0.5")
    )
    MEDIAPIPE_HAND_MIN_TRACKING_CONFIDENCE = float(
        os.getenv("MEDIAPIPE_HAND_MIN_TRACKING_CONFIDENCE", "0.5")
    )

    # Vocal Tone model: directory containing vocal_tone_model.pkl, vocal_tone_scaler.pkl, vocal_tone_labels.pkl
    # If set, overrides default backend/models/vocal_tone (e.g. /data/models/vocal_tone if mounted)
    VOCAL_TONE_MODEL_DIR = os.getenv("VOCAL_TONE_MODEL_DIR", "")
    # Google Drive: file ID of vocal_tone_model.zip (share link "Anyone with the link can view")
    # If set, entrypoint will download and extract to models/vocal_tone when files are missing
    DRIVE_VOCAL_TONE_MODEL_ZIP_ID = os.getenv("DRIVE_VOCAL_TONE_MODEL_ZIP_ID", "")

    # SAVEE dataset path for Vocal Tone training (folder with anger/, disgust/, fear/, etc.)
    # When running locally (not in Docker), set this to your local path so train_model.py finds the data
    SAVEE_DATASET_PATH = os.getenv("SAVEE_DATASET_PATH", "")

    # Clifton Fusion model configuration
    # Threshold below which a domain is considered a development opportunity
    # Default: 0.15 (below 60% of even distribution across 4 domains)
    CLIFTON_DEVELOPMENT_THRESHOLD = float(
        os.getenv("CLIFTON_DEVELOPMENT_THRESHOLD", "0.15")
    )

    # Ollama interviewer questions model
    # Base URL of the Ollama server
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    # Name of the Ollama model to use (e.g. llama3, mistral, phi3:mini, gemma3:270m, etc.)
    # Default to the very small \"gemma3:270m\" model to fit low-RAM environments.
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma3:1b")
    # Timeout in seconds for Ollama HTTP requests
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
