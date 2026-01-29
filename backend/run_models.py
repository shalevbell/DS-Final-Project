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
import librosa
import soundfile as sf
import joblib

from config import Config

logger = logging.getLogger(__name__)

# Path to SAVEE dataset for Vocal Tone training
# Windows path (when running locally on Windows)
SAVEE_DATASET_PATH_WINDOWS = r"C:\Users\shira\OneDrive\שולחן העבודה\finalProjectFiles\DB for Proj\Savee-Classifier"
# Linux path (when running inside Docker container)
# Note: docker-compose maps the parent directory, so we need to go into Savee-Classifier
SAVEE_DATASET_PATH_LINUX = "/data/dataset"
# Default - will be determined at runtime
SAVEE_DATASET_PATH = None

def get_available_models():
    """
    Get list of available analysis models.

    Returns:
        List of model names that can be used for chunk analysis
    """
    # MODEL_REGISTRY is defined after functions but will be available at runtime
    return list(MODEL_REGISTRY.keys())


def list_savee_dataset_files(dataset_path: str = None) -> Dict:
    """
    List all files in the SAVEE dataset directory for Vocal Tone model training.

    Args:
        dataset_path: Path to the SAVEE dataset directory. 
                     If None, uses the default path (detected automatically).

    Returns:
        Dictionary with file listing results:
        {
            'path': str,
            'exists': bool,
            'total_files': int,
            'files': list of file info dicts,
            'directories': list of subdirectory names,
            'file_extensions': dict with counts by extension
        }
    """
    global SAVEE_DATASET_PATH
    
    if dataset_path is None:
        # Auto-detect which path to use
        if SAVEE_DATASET_PATH is None:
            # Check if running in Docker (Linux path)
            linux_path = Path(SAVEE_DATASET_PATH_LINUX)
            if linux_path.exists():
                SAVEE_DATASET_PATH = SAVEE_DATASET_PATH_LINUX
            else:
                # Check Windows path
                windows_path = Path(SAVEE_DATASET_PATH_WINDOWS)
                if windows_path.exists():
                    SAVEE_DATASET_PATH = SAVEE_DATASET_PATH_WINDOWS
                else:
                    # Default to Linux path (Docker)
                    SAVEE_DATASET_PATH = SAVEE_DATASET_PATH_LINUX
        
        dataset_path = SAVEE_DATASET_PATH
    
    dataset_path_obj = Path(dataset_path)
    
    result = {
        'path': str(dataset_path_obj),
        'exists': False,
        'total_files': 0,
        'files': [],
        'directories': [],
        'file_extensions': {},
        'error': None
    }
    
    try:
        if not dataset_path_obj.exists():
            error_msg = f'Dataset path does not exist: {dataset_path}'
            logger.error(f'[VocalTone] {error_msg}')
            result['error'] = error_msg
            return result
        
        if not dataset_path_obj.is_dir():
            error_msg = f'Path is not a directory: {dataset_path}'
            logger.error(f'[VocalTone] {error_msg}')
            result['error'] = error_msg
            return result
        
        result['exists'] = True
        
        # List all files and directories
        files_list = []
        directories_list = []
        extensions_count = {}
        
        for item in dataset_path_obj.iterdir():
            if item.is_file():
                file_info = {
                    'name': item.name,
                    'path': str(item),
                    'size_bytes': item.stat().st_size,
                    'size_mb': round(item.stat().st_size / (1024 * 1024), 2),
                    'extension': item.suffix.lower()
                }
                files_list.append(file_info)
                
                # Count extensions
                ext = item.suffix.lower() or 'no_extension'
                extensions_count[ext] = extensions_count.get(ext, 0) + 1
            
            elif item.is_dir():
                directories_list.append(item.name)
        
        # Sort files by name
        files_list.sort(key=lambda x: x['name'])
        directories_list.sort()
        
        result['total_files'] = len(files_list)
        result['files'] = files_list
        result['directories'] = directories_list
        result['file_extensions'] = extensions_count
        
        # Print to console/logs
        logger.info(f'[VocalTone] Dataset path: {dataset_path}')
        logger.info(f'[VocalTone] Total files found: {result["total_files"]}')
        logger.info(f'[VocalTone] Directories: {len(directories_list)}')
        logger.info(f'[VocalTone] File extensions: {extensions_count}')
        
        if files_list:
            logger.info('[VocalTone] Sample files (first 10):')
            for file_info in files_list[:10]:
                logger.info(f'  - {file_info["name"]} ({file_info["size_mb"]} MB, {file_info["extension"]})')
            if len(files_list) > 10:
                logger.info(f'  ... and {len(files_list) - 10} more files')
        
        if directories_list:
            logger.info(f'[VocalTone] Subdirectories: {", ".join(directories_list)}')
        
        return result
        
    except PermissionError as e:
        error_msg = f'Permission denied accessing path: {dataset_path}'
        logger.error(f'[VocalTone] {error_msg}: {e}')
        result['error'] = error_msg
        return result
    except Exception as e:
        error_msg = f'Error listing dataset files: {str(e)}'
        logger.error(f'[VocalTone] {error_msg}\n{traceback.format_exc()}')
        result['error'] = error_msg
        return result


def augment_audio(audio: np.ndarray, sr: int, augmentation_type: str = 'none') -> np.ndarray:
    """
    Apply data augmentation to audio signal.
    
    Args:
        audio: Audio waveform array
        sr: Sample rate
        augmentation_type: Type of augmentation:
            - 'none': No augmentation (original)
            - 'time_stretch': Time stretching (±10%)
            - 'pitch_shift': Pitch shifting (±2 semitones)
            - 'noise': Add Gaussian noise (SNR ~20dB)
            - 'time_shift': Time shifting (circular shift)
    
    Returns:
        Augmented audio array (same length as input)
    """
    if augmentation_type == 'none':
        return audio
    
    elif augmentation_type == 'time_stretch':
        # Time stretching: speed up or slow down by ±10%
        rate = np.random.uniform(0.9, 1.1)
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        # Crop or pad to original length
        target_len = len(audio)
        if len(stretched) > target_len:
            stretched = stretched[:target_len]
        elif len(stretched) < target_len:
            stretched = np.pad(stretched, (0, target_len - len(stretched)), mode='constant')
        return stretched
    
    elif augmentation_type == 'pitch_shift':
        # Pitch shifting: shift by ±2 semitones
        n_steps = np.random.uniform(-2, 2)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    elif augmentation_type == 'noise':
        # Add Gaussian noise (SNR ~20dB)
        noise_level = np.random.uniform(0.01, 0.05)  # 1-5% of signal amplitude
        noise = np.random.normal(0, noise_level, len(audio))
        augmented = audio + noise
        # Re-normalize to prevent clipping
        max_val = np.abs(augmented).max()
        if max_val > 0:
            augmented = augmented / max_val * 0.95
        return augmented
    
    elif augmentation_type == 'time_shift':
        # Circular time shift (wrap around)
        shift_amount = np.random.randint(0, len(audio))
        return np.roll(audio, shift_amount)
    
    else:
        return audio


def extract_extended_features(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 40
) -> np.ndarray:
    """
    Extract extended feature set from audio:
    - MFCC features (mean + std): 40 * 2 = 80 features
    - Chroma features (mean + std): 12 * 2 = 24 features
    - Spectral features (centroid, rolloff, zero-crossing rate): 3 features
    
    Total: 80 + 24 + 3 = 107 features
    
    Args:
        audio: Audio waveform array
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients (default: 40)
    
    Returns:
        Feature vector of length 107
    """
    features = []
    
    # 1. MFCC features (80 features: 40 mean + 40 std)
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    features.extend(mfcc_mean)
    features.extend(mfcc_std)
    
    # 2. Chroma features (24 features: 12 mean + 12 std)
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512
    )
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    features.extend(chroma_mean)
    features.extend(chroma_std)
    
    # 3. Spectral features (3 features)
    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features.append(np.mean(spectral_centroids))
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features.append(np.mean(spectral_rolloff))
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features.append(np.mean(zcr))
    
    return np.array(features)


def process_savee_dataset_for_training(
    dataset_path: str = None,
    target_sr: int = 16000,
    target_duration_sec: float = 3.0,
    n_mfcc: int = 40,
    show_progress: bool = True,
    use_augmentation: bool = True,
    augmentation_factor: int = 2
) -> tuple:
    """
    Process all WAV files in SAVEE dataset for Vocal Tone model training.
    
    For each WAV file:
    - Load audio waveform
    - Normalize format: mono + 16kHz + light normalization
    - Fix length: crop or pad to target_duration_sec (default 3 seconds)
    - Extract extended features: MFCC + Chroma + Spectral (107 features total)
    - Optionally apply data augmentation to increase dataset size
    
    Args:
        dataset_path: Path to Savee-Classifier directory. If None, uses default path.
        target_sr: Target sample rate (default: 16000 Hz)
        target_duration_sec: Target duration in seconds (default: 3.0)
        n_mfcc: Number of MFCC coefficients to extract (default: 40)
        show_progress: Whether to print progress messages
        use_augmentation: Whether to apply data augmentation (default: True)
        augmentation_factor: Number of augmented versions per original file (default: 2)
    
    Returns:
        Tuple of (X, y, labels_map) where:
        - X: numpy array of shape (#samples, #features) where #features = 107
        - y: numpy array of shape (#samples,) with label indices
        - labels_map: dict mapping label index to label name (directory name)
    """
    global SAVEE_DATASET_PATH
    
    if dataset_path is None:
        # Auto-detect path
        if SAVEE_DATASET_PATH is None:
            linux_path = Path(SAVEE_DATASET_PATH_LINUX)
            if linux_path.exists():
                SAVEE_DATASET_PATH = SAVEE_DATASET_PATH_LINUX
            else:
                windows_path = Path(SAVEE_DATASET_PATH_WINDOWS)
                if windows_path.exists():
                    SAVEE_DATASET_PATH = SAVEE_DATASET_PATH_WINDOWS
                else:
                    SAVEE_DATASET_PATH = SAVEE_DATASET_PATH_LINUX
        dataset_path = SAVEE_DATASET_PATH
    
    dataset_path_obj = Path(dataset_path)
    
    if not dataset_path_obj.exists():
        raise ValueError(f'Dataset path does not exist: {dataset_path}')
    
    if not dataset_path_obj.is_dir():
        raise ValueError(f'Path is not a directory: {dataset_path}')
    
    # Find all WAV files organized by subdirectory (each subdirectory is a label)
    wav_files = []
    labels_list = []
    
    if show_progress:
        logger.info(f'[VocalTone] Scanning dataset: {dataset_path}')
    
    # Get all subdirectories (each is a label/class)
    subdirs = [d for d in dataset_path_obj.iterdir() if d.is_dir()]
    subdirs.sort()
    
    labels_map = {}
    label_to_idx = {}
    
    for subdir in subdirs:
        label_name = subdir.name
        if label_name not in label_to_idx:
            idx = len(label_to_idx)
            label_to_idx[label_name] = idx
            labels_map[idx] = label_name
        
        # Find all WAV files in this subdirectory
        wav_files_in_dir = list(subdir.glob('*.wav')) + list(subdir.glob('*.WAV'))
        
        for wav_file in wav_files_in_dir:
            wav_files.append(wav_file)
            labels_list.append(label_to_idx[label_name])
    
    if len(wav_files) == 0:
        raise ValueError(f'No WAV files found in dataset: {dataset_path}')
    
    if show_progress:
        logger.info(f'[VocalTone] Found {len(wav_files)} WAV files in {len(label_to_idx)} classes')
        logger.info(f'[VocalTone] Classes: {list(label_to_idx.keys())}')
    
    # Process each WAV file
    features_list = []
    valid_labels = []
    
    target_samples = int(target_duration_sec * target_sr)
    
    # Augmentation types to apply
    augmentation_types = ['none']  # Always include original
    if use_augmentation:
        augmentation_types.extend(['time_stretch', 'pitch_shift', 'noise', 'time_shift'])
        # Limit augmentation to augmentation_factor per file
        augmentation_types = augmentation_types[:augmentation_factor + 1]
    
    total_files_to_process = len(wav_files) * len(augmentation_types)
    processed_count = 0
    
    if show_progress:
        print(f"   📊 Processing {len(wav_files)} files × {len(augmentation_types)} versions = {total_files_to_process} total samples")
        print(f"   🔄 Augmentation types: {augmentation_types}")
        print(f"   ⏳ This may take 15-30 minutes...")
        print()
    
    for i, (wav_path, label_idx) in enumerate(zip(wav_files, labels_list)):
        try:
            if show_progress:
                # Show progress more frequently (every 10 files or every 25 samples)
                if (i + 1) % 10 == 0 or processed_count % 25 == 0:
                    progress_pct = (i + 1) / len(wav_files) * 100
                    print(f"   Progress: {i + 1}/{len(wav_files)} files ({progress_pct:.1f}%) | Samples created: {processed_count}/{total_files_to_process}")
            
            # Load audio (force mono)
            audio, sr = librosa.load(str(wav_path), sr=None, mono=True)
            
            # Resample to target sample rate if needed
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
            # Light normalization (divide by max absolute value, cap at 0.95 to avoid clipping)
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val * 0.95
            
            # Fix length: crop or pad to target_duration_sec
            current_samples = len(audio)
            if current_samples > target_samples:
                # Crop: take first target_samples
                audio = audio[:target_samples]
            elif current_samples < target_samples:
                # Pad: zero-padding at the end
                audio = np.pad(audio, (0, target_samples - current_samples), mode='constant')
            
            # Process original + augmented versions
            for aug_type in augmentation_types:
                # Apply augmentation (if not 'none')
                augmented_audio = augment_audio(audio.copy(), target_sr, aug_type)
                
                # Ensure length is still correct after augmentation
                if len(augmented_audio) != target_samples:
                    if len(augmented_audio) > target_samples:
                        augmented_audio = augmented_audio[:target_samples]
                    else:
                        augmented_audio = np.pad(augmented_audio, (0, target_samples - len(augmented_audio)), mode='constant')
                
                # Extract extended features (MFCC + Chroma + Spectral)
                feature_vector = extract_extended_features(augmented_audio, target_sr, n_mfcc)
                
                features_list.append(feature_vector)
                valid_labels.append(label_idx)
                processed_count += 1
            
        except Exception as e:
            if show_progress:
                print(f"   ⚠️  Warning: Error processing {wav_path.name}: {e}")
            logger.warning(f'[VocalTone] Error processing {wav_path.name}: {e}')
            continue
    
    if len(features_list) == 0:
        raise ValueError('No valid audio files processed')
    
    # Convert to numpy arrays
    X = np.array(features_list)  # Shape: (#samples, 80)
    y = np.array(valid_labels)   # Shape: (#samples,)
    
    if show_progress:
        print()
        print(f"   ✅ Processing complete!")
        print(f"   📊 X shape: {X.shape} (#samples, #features)")
        print(f"   📊 y shape: {y.shape} (#samples,)")
        print(f"   📊 Features per sample: {X.shape[1]} (extended: MFCC + Chroma + Spectral)")
        print(f"   📊 Number of classes: {len(labels_map)}")
        if use_augmentation:
            print(f"   📊 Dataset size increased from {len(wav_files)} to {X.shape[0]} samples (×{X.shape[0]/len(wav_files):.1f})")
        logger.info(f'[VocalTone] Processing complete!')
        logger.info(f'[VocalTone] X shape: {X.shape} (#samples, #features)')
        logger.info(f'[VocalTone] y shape: {y.shape} (#samples,)')
        logger.info(f'[VocalTone] Features per sample: {X.shape[1]} (extended: MFCC + Chroma + Spectral)')
        logger.info(f'[VocalTone] Number of classes: {len(labels_map)}')
        logger.info(f'[VocalTone] Samples per class:')
        for label_idx, label_name in labels_map.items():
            count = np.sum(y == label_idx)
            logger.info(f'  {label_name}: {count} samples')
    
    return X, y, labels_map


def analyze_audio_whisper(audio_bytes: bytes, session_id: str, chunk_index: int) -> Dict:
    """
    Transcribe audio using Whisper.

    TODO: Integrate actual Whisper model from /test/whisper_test.py

    Args:
        audio_bytes: WAV audio data
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
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
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

                # Process with landmarkers (IMAGE mode - no timestamp needed)
                try:
                    face_result = analyze_video_mediapipe._landmarkers['face'].detect(mp_image)
                    pose_result = analyze_video_mediapipe._landmarkers['pose'].detect(mp_image)
                    hand_result = analyze_video_mediapipe._landmarkers['hand'].detect(mp_image)

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

    Uses the trained Vocal Tone model to predict emotion from audio features.

    Args:
        audio_bytes: WAV audio data
        session_id: Session identifier
        chunk_index: Chunk index within session

    Returns:
        Dictionary with vocal analysis results:
        {
            'emotion': str,
            'confidence': float,
            'emotion_probabilities': dict,
            'pitch_mean': float,
            'pitch_std': float,
            'tempo': float,
            'energy_level': float,
            'processing_time_ms': int
        }
    """
    start_time = time.time()
    temp_path = None

    try:
        logger.info(f'[VocalTone] Processing chunk {session_id}:{chunk_index} ({len(audio_bytes)} bytes)')

        # Create temporary WAV file from audio_bytes (same pattern as Whisper)
        if isinstance(audio_bytes, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                temp_path = tmp_file.name
            logger.info(
                f'[VocalTone] Saved audio to temp file: {temp_path} '
                f'({len(audio_bytes)} bytes)'
            )
            audio_source = temp_path
        elif isinstance(audio_bytes, str):
            audio_source = audio_bytes
            logger.info(f'[VocalTone] Using audio path: {audio_source}')
        else:
            raise ValueError(f'Unsupported audio input type: {type(audio_bytes).__name__}')

        # Load model files (lazy loading - only load once)
        if not hasattr(analyze_vocal_tone, '_model'):
            logger.info('[VocalTone] Loading model files (first use)...')
            
            # Determine backend directory path
            backend_dir = Path(__file__).parent
            models_dir = backend_dir / 'models' / 'vocal_tone'
            model_path = models_dir / 'vocal_tone_model.pkl'
            scaler_path = models_dir / 'vocal_tone_scaler.pkl'
            labels_path = models_dir / 'vocal_tone_labels.pkl'

            if not model_path.exists():
                error_msg = f'Vocal Tone model not found: {model_path}. Please train the model first.'
                logger.error(f'[VocalTone] {error_msg}')
                return {
                    'error': error_msg,
                    'processing_time_ms': int((time.time() - start_time) * 1000)
                }

            try:
                analyze_vocal_tone._model = joblib.load(model_path)
                analyze_vocal_tone._scaler = joblib.load(scaler_path)
                analyze_vocal_tone._labels_map = joblib.load(labels_path)
                logger.info(f'[VocalTone] Model loaded: {type(analyze_vocal_tone._model).__name__}')
                logger.info(f'[VocalTone] Labels: {len(analyze_vocal_tone._labels_map)} classes')
            except Exception as e:
                error_msg = f'Failed to load Vocal Tone model: {e}'
                logger.error(f'[VocalTone] {error_msg}')
                return {
                    'error': error_msg,
                    'processing_time_ms': int((time.time() - start_time) * 1000)
                }

        model = analyze_vocal_tone._model
        scaler = analyze_vocal_tone._scaler
        labels_map = analyze_vocal_tone._labels_map

        # Process audio file: extract features (same as training)
        target_sr = 16000
        target_duration_sec = 3.0
        n_mfcc = 40

        logger.info(f'[VocalTone] Loading and processing audio: {audio_source}')
        
        # Load audio (force mono)
        audio, sr = librosa.load(audio_source, sr=None, mono=True)
        
        # Resample to target sample rate if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Light normalization
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        # Fix length: crop or pad to target_duration_sec
        target_samples = int(target_duration_sec * target_sr)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        
        # Extract extended features (MFCC + Chroma + Spectral)
        feature_vector = extract_extended_features(audio, target_sr, n_mfcc)
        
        logger.info(f'[VocalTone] Features extracted: shape {feature_vector.shape}')

        # Scale features
        features_scaled = scaler.transform([feature_vector])

        # Predict emotion
        prediction_idx = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        predicted_emotion = labels_map[prediction_idx]
        confidence = probabilities[prediction_idx]

        # Calculate additional audio features for richer output
        # Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=audio, sr=target_sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        pitch_mean = float(np.mean(pitch_values)) if pitch_values else 0.0
        pitch_std = float(np.std(pitch_values)) if pitch_values else 0.0

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=target_sr)
        tempo_value = float(tempo) if tempo is not None else 0.0

        # Energy level
        energy = float(np.mean(librosa.feature.rms(y=audio)[0]))

        # Build emotion probabilities dictionary
        emotion_probs = {
            labels_map[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }

        processing_time = int((time.time() - start_time) * 1000)

        result = {
            'emotion': predicted_emotion,
            'confidence': float(confidence),
            'emotion_probabilities': emotion_probs,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'tempo': tempo_value,
            'energy_level': energy,
            'processing_time_ms': processing_time
        }

        logger.info(
            f'[VocalTone] Completed chunk {session_id}:{chunk_index} in {processing_time}ms: '
            f'emotion={predicted_emotion} (conf={confidence:.3f}), '
            f'pitch={pitch_mean:.1f}Hz, tempo={tempo_value:.1f}BPM'
        )

        return result

    except Exception as e:
        logger.error(
            f'[VocalTone] Error processing chunk {session_id}:{chunk_index}: {e}\n'
            f'{traceback.format_exc()}'
        )
        return {
            'error': str(e),
            'processing_time_ms': int((time.time() - start_time) * 1000)
        }
    finally:
        # Clean up temp file (same pattern as Whisper)
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f'[VocalTone] Removed temp file: {temp_path}')
            except Exception as cleanup_error:
                logger.warning(f'[VocalTone] Temp cleanup failed: {cleanup_error}')


# =============================================================================
# SUMMARY LOGGING
# =============================================================================

def log_chunk_summary(session_id: str, chunk_index: int, results: Dict):
    """
    Print a formatted summary of all model results for a chunk.
    
    Args:
        session_id: Session identifier
        chunk_index: Chunk index
        results: Dictionary with all model results
    """
    logger.info('')
    logger.info('=' * 80)
    logger.info(f'📊 CHUNK SUMMARY: {session_id}:{chunk_index}')
    logger.info('=' * 80)
    
    # Whisper results
    if 'whisper' in results:
        whisper_result = results['whisper']
        if 'error' in whisper_result:
            logger.info('🎤 WHISPER (Transcription): ❌ FAILED')
            logger.info(f'   Error: {whisper_result["error"]}')
        else:
            transcript = whisper_result.get('transcript', '')
            language = whisper_result.get('language', 'unknown')
            confidence = whisper_result.get('confidence')
            processing_time = whisper_result.get('processing_time_ms', 0)
            
            logger.info('🎤 WHISPER (Transcription): ✅ SUCCESS')
            if transcript:
                # Truncate long transcripts for readability
                display_transcript = transcript[:200] + '...' if len(transcript) > 200 else transcript
                logger.info(f'   Transcript: "{display_transcript}"')
            else:
                logger.info('   Transcript: (no speech detected)')
            logger.info(f'   Language: {language}')
            if confidence is not None:
                logger.info(f'   Confidence: {confidence:.3f}')
            logger.info(f'   Processing time: {processing_time}ms')
    
    # MediaPipe results
    if 'mediapipe' in results:
        mediapipe_result = results['mediapipe']
        if 'error' in mediapipe_result:
            logger.info('📹 MEDIAPIPE (Video Analysis): ❌ FAILED')
            logger.info(f'   Error: {mediapipe_result["error"]}')
        else:
            emotion = mediapipe_result.get('dominant_emotion', 'unknown')
            emotion_conf = mediapipe_result.get('emotion_confidence', 0)
            posture = mediapipe_result.get('posture_score', 0)
            engagement = mediapipe_result.get('engagement_score', 0)
            hands = mediapipe_result.get('hand_gestures_detected', False)
            frames = mediapipe_result.get('frames_analyzed', 0)
            processing_time = mediapipe_result.get('processing_time_ms', 0)
            
            logger.info('📹 MEDIAPIPE (Video Analysis): ✅ SUCCESS')
            logger.info(f'   Dominant Emotion: {emotion} (confidence: {emotion_conf:.2f})')
            logger.info(f'   Posture Score: {posture:.2f}')
            logger.info(f'   Engagement Score: {engagement:.2f}')
            logger.info(f'   Hand Gestures: {"Yes" if hands else "No"}')
            logger.info(f'   Frames Analyzed: {frames}')
            logger.info(f'   Processing time: {processing_time}ms')
    
    # VocalTone results
    if 'vocaltone' in results:
        vocaltone_result = results['vocaltone']
        if 'error' in vocaltone_result:
            logger.info('🎵 VOCAL TONE (Audio Emotion): ❌ FAILED')
            logger.info(f'   Error: {vocaltone_result["error"]}')
        else:
            emotion = vocaltone_result.get('emotion', 'unknown')
            confidence = vocaltone_result.get('confidence', 0)
            pitch_mean = vocaltone_result.get('pitch_mean', 0)
            tempo = vocaltone_result.get('tempo', 0)
            energy = vocaltone_result.get('energy_level', 0)
            processing_time = vocaltone_result.get('processing_time_ms', 0)
            
            logger.info('🎵 VOCAL TONE (Audio Emotion): ✅ SUCCESS')
            logger.info(f'   Predicted Emotion: {emotion} (confidence: {confidence:.3f})')
            logger.info(f'   Pitch Mean: {pitch_mean:.1f} Hz')
            logger.info(f'   Tempo: {tempo:.1f} BPM')
            logger.info(f'   Energy Level: {energy:.3f}')
            logger.info(f'   Processing time: {processing_time}ms')
    
    # Overall summary
    total_time = results.get('total_processing_time_ms', 0)
    # Count successful models (exclude total_processing_time_ms which is an int)
    model_keys = [k for k in results.keys() if k != 'total_processing_time_ms']
    success_count = sum(1 for k in model_keys 
                       if isinstance(results[k], dict) and 'error' not in results[k])
    total_models = len(model_keys)
    
    logger.info('')
    logger.info('─' * 80)
    logger.info(f'⏱️  Total Processing Time: {total_time}ms ({total_time/1000:.2f}s)')
    logger.info(f'✅ Success Rate: {success_count}/{total_models} models succeeded')
    logger.info('=' * 80)
    logger.info('')


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
        audio_bytes: WAV audio data
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

    # Calculate success count (exclude total_processing_time_ms which is an int)
    model_keys = [k for k in results.keys() if k != 'total_processing_time_ms']
    success_count = sum(1 for k in model_keys 
                       if isinstance(results[k], dict) and 'error' not in results[k])
    total_models = len(model_keys)
    logger.info(
        f'Parallel analysis complete for chunk {session_id}:{chunk_index}: '
        f'{success_count}/{total_models} succeeded in {total_time}ms'
    )

    # Print formatted summary
    log_chunk_summary(session_id, chunk_index, results)

    return results
