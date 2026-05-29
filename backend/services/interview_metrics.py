"""
Per-session baseline calibration and derived interview metrics.

Computes Speaking Rate (WPM + classification) and Stress Level (0-100%)
relative to each candidate's baseline from the first processed chunk.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Speaking rate bands relative to personal baseline WPM
SPEAKING_RATE_SLOW_RATIO = 0.85
SPEAKING_RATE_FAST_RATIO = 1.15

# Spike magnitudes treated as maximum stress contribution (100% of sub-score)
ACOUSTIC_SPIKE_CAP = 1.0
VISUAL_MOVEMENT_SPIKE_CAP = 1.0

# MediaPipe face emotions that increase visual stress
STRESS_VISUAL_EMOTIONS = frozenset(
    {
        "frustrated",
        "concerned",
        "confused",
        "sad",
        "fear",
        "nervous",
        "angry",
    }
)

# VocalTone / SAVEE-style labels that increase visual+acoustic stress context
STRESS_VOCAL_EMOTIONS = frozenset(
    {
        "anger",
        "angry",
        "disgust",
        "fear",
        "fearful",
        "sadness",
        "sad",
        "nervous",
        "stressed",
    }
)


@dataclass
class SessionBaseline:
    """Calibrated baseline metrics for one interview session."""

    baseline_wpm: Optional[float] = None
    baseline_pitch_std: Optional[float] = None
    baseline_energy_level: Optional[float] = None
    baseline_movement_variance: Optional[float] = None
    established: bool = False
    calibration_chunk_index: Optional[int] = None


class BaselineStateManager:
    """
    Thread-safe store of per-session baselines.

    The first chunk with valid metric inputs establishes the baseline.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionBaseline] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> SessionBaseline:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionBaseline()
            return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def establish_if_needed(
        self,
        session_id: str,
        chunk_index: int,
        *,
        wpm: Optional[float],
        pitch_std: Optional[float],
        energy_level: Optional[float],
        movement_variance: Optional[float],
    ) -> SessionBaseline:
        """
        Record baseline values from the first chunk that supplies usable data.

        Fills any missing baseline fields; marks established once WPM or both
        acoustic and visual baselines are available.
        """
        baseline = self.get(session_id)
        with self._lock:
            if baseline.established:
                return baseline

            if baseline.calibration_chunk_index is None:
                baseline.calibration_chunk_index = chunk_index

            if wpm is not None and wpm > 0 and baseline.baseline_wpm is None:
                baseline.baseline_wpm = float(wpm)
            if pitch_std is not None and pitch_std >= 0 and baseline.baseline_pitch_std is None:
                baseline.baseline_pitch_std = float(pitch_std)
            if energy_level is not None and energy_level >= 0 and baseline.baseline_energy_level is None:
                baseline.baseline_energy_level = float(energy_level)
            if (
                movement_variance is not None
                and movement_variance >= 0
                and baseline.baseline_movement_variance is None
            ):
                baseline.baseline_movement_variance = float(movement_variance)

            has_wpm = baseline.baseline_wpm is not None
            has_acoustic = (
                baseline.baseline_pitch_std is not None
                and baseline.baseline_energy_level is not None
            )
            has_visual = baseline.baseline_movement_variance is not None

            if has_wpm or (has_acoustic and has_visual):
                baseline.established = True
                logger.info(
                    "[Baseline] Established for session=%s chunk=%s wpm=%s "
                    "pitch_std=%s energy=%s movement_var=%s",
                    session_id,
                    baseline.calibration_chunk_index,
                    baseline.baseline_wpm,
                    baseline.baseline_pitch_std,
                    baseline.baseline_energy_level,
                    baseline.baseline_movement_variance,
                )
            return baseline


# Process-wide singleton (chunk workers share this within one backend process)
baseline_manager = BaselineStateManager()


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _relative_spike(current: float, baseline: Optional[float], cap: float = ACOUSTIC_SPIKE_CAP) -> float:
    """
    Fractional increase over baseline, clamped to [0, cap].

    Example: baseline=10, current=15 -> spike=0.5 (50% increase).
    """
    if baseline is None or baseline <= 1e-9:
        return 0.0
    spike = (current - baseline) / baseline
    if spike < 0:
        return 0.0
    return min(spike, cap)


def _spike_to_unit(spike: float, cap: float) -> float:
    """Map spike in [0, cap] to unit score in [0, 1]."""
    if cap <= 0:
        return 0.0
    return min(max(spike, 0.0) / cap, 1.0)


def _classify_speaking_rate(ratio: float) -> str:
    if ratio < SPEAKING_RATE_SLOW_RATIO:
        return "Slow"
    if ratio > SPEAKING_RATE_FAST_RATIO:
        return "Fast"
    return "Normal"


def _negative_emotion_score(
    mediapipe_data: Dict[str, Any],
    vocaltone_data: Dict[str, Any],
) -> float:
    """
    Visual stress proxy from dominant emotions (0..1).

    Uses both MediaPipe and VocalTone labels when available.
    """
    scores: List[float] = []

    mp_emotion = (mediapipe_data.get("dominant_emotion") or "").strip().lower()
    if mp_emotion in STRESS_VISUAL_EMOTIONS:
        conf = _safe_float(mediapipe_data.get("emotion_confidence"), 0.7) or 0.7
        scores.append(min(max(conf, 0.0), 1.0))

    vt_emotion = (vocaltone_data.get("emotion") or "").strip().lower()
    if vt_emotion in STRESS_VOCAL_EMOTIONS:
        conf = _safe_float(vocaltone_data.get("confidence"), 0.7) or 0.7
        scores.append(min(max(conf, 0.0), 1.0))

    # Boost if VocalTone assigns high probability to a stress class
    probs = vocaltone_data.get("emotion_probabilities") or {}
    if isinstance(probs, dict):
        for label, prob in probs.items():
            if str(label).lower() in STRESS_VOCAL_EMOTIONS:
                p = _safe_float(prob, 0.0) or 0.0
                if p >= 0.35:
                    scores.append(min(p, 1.0))

    if not scores:
        return 0.0
    return max(scores)


def calculate_speaking_rate(
    whisper_data: Dict[str, Any],
    chunk_duration_sec: float,
    session_id: str,
    chunk_index: int,
) -> Dict[str, Any]:
    """
    Compute words-per-minute and Slow/Normal/Fast vs personal baseline.

    Args:
        whisper_data: Output dict from analyze_audio_whisper
        chunk_duration_sec: Duration of the chunk in seconds
        session_id: Interview session id
        chunk_index: Zero-based chunk index

    Returns:
        Dict with wpm, classification, baseline fields, and ratio.
    """
    if whisper_data.get("error"):
        return {"error": whisper_data["error"]}

    word_count = int(whisper_data.get("word_count") or 0)
    duration = max(float(chunk_duration_sec or 0), 1e-3)
    wpm = (word_count / duration) * 60.0

    baseline = baseline_manager.get(session_id)

    if not baseline.established or baseline.baseline_wpm is None:
        return {
            "wpm": round(wpm, 2),
            "word_count": word_count,
            "chunk_duration_sec": round(duration, 3),
            "classification": "Normal",
            "baseline_wpm": round(wpm, 2) if wpm > 0 else None,
            "ratio_to_baseline": 1.0,
            "is_baseline_chunk": True,
        }

    base_wpm = baseline.baseline_wpm
    ratio = wpm / base_wpm if base_wpm and base_wpm > 0 else 1.0
    is_baseline_chunk = baseline.calibration_chunk_index == chunk_index

    return {
        "wpm": round(wpm, 2),
        "word_count": word_count,
        "chunk_duration_sec": round(duration, 3),
        "classification": _classify_speaking_rate(ratio),
        "baseline_wpm": round(base_wpm, 2),
        "ratio_to_baseline": round(ratio, 3),
        "is_baseline_chunk": is_baseline_chunk,
    }


def calculate_stress_level(
    vocaltone_data: Dict[str, Any],
    mediapipe_data: Dict[str, Any],
    session_id: str,
    chunk_index: int,
) -> Dict[str, Any]:
    """
    Compute stress score (0-100) from acoustic and visual deviation vs baseline.

    Acoustic (50%): pitch_std and energy_level spikes vs baseline.
    Visual (50%): negative emotions + movement_variance spike vs baseline.
    """
    if vocaltone_data.get("error") and mediapipe_data.get("error"):
        return {
            "stress_percent": None,
            "error": "Both vocaltone and mediapipe failed",
        }

    pitch_std = _safe_float(vocaltone_data.get("pitch_std"), 0.0) or 0.0
    energy_level = _safe_float(vocaltone_data.get("energy_level"), 0.0) or 0.0
    movement_variance = _safe_float(mediapipe_data.get("movement_variance"), 0.0) or 0.0
    posture_delta = _safe_float(mediapipe_data.get("posture_delta"), 0.0) or 0.0

    baseline = baseline_manager.get(session_id)

    if not baseline.established:
        return {
            "stress_percent": 0.0,
            "acoustic_stress": 0.0,
            "visual_stress": 0.0,
            "is_baseline_chunk": True,
            "components": {
                "pitch_std_spike": 0.0,
                "energy_spike": 0.0,
                "negative_emotion": 0.0,
                "movement_spike": 0.0,
            },
        }

    pitch_spike = _relative_spike(
        pitch_std,
        baseline.baseline_pitch_std,
        cap=ACOUSTIC_SPIKE_CAP,
    )
    energy_spike = _relative_spike(
        energy_level,
        baseline.baseline_energy_level,
        cap=ACOUSTIC_SPIKE_CAP,
    )
    movement_spike = _relative_spike(
        movement_variance,
        baseline.baseline_movement_variance,
        cap=VISUAL_MOVEMENT_SPIKE_CAP,
    )

    pitch_score = _spike_to_unit(pitch_spike, ACOUSTIC_SPIKE_CAP)
    energy_score = _spike_to_unit(energy_spike, ACOUSTIC_SPIKE_CAP)
    movement_score = _spike_to_unit(movement_spike, VISUAL_MOVEMENT_SPIKE_CAP)
    emotion_score = _negative_emotion_score(mediapipe_data, vocaltone_data)

    acoustic_stress = 0.5 * pitch_score + 0.5 * energy_score
    visual_stress = 0.5 * emotion_score + 0.5 * movement_score
    stress_unit = 0.5 * acoustic_stress + 0.5 * visual_stress
    stress_percent = round(min(max(stress_unit * 100.0, 0.0), 100.0), 1)

    return {
        "stress_percent": stress_percent,
        "acoustic_stress": round(acoustic_stress * 100.0, 1),
        "visual_stress": round(visual_stress * 100.0, 1),
        "is_baseline_chunk": baseline.calibration_chunk_index == chunk_index,
        "components": {
            "pitch_std_spike": round(pitch_spike, 3),
            "energy_spike": round(energy_spike, 3),
            "negative_emotion": round(emotion_score, 3),
            "movement_spike": round(movement_spike, 3),
        },
        "posture_delta": round(posture_delta, 5),
        "movement_variance": round(movement_variance, 6),
    }


def attach_interview_metrics(
    results: Dict[str, Any],
    session_id: str,
    chunk_index: int,
    on_metrics_ready: Optional[Callable[[str, int, Dict[str, Any]], None]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute interview metrics from model outputs and attach to results dict.

    Invokes on_metrics_ready(session_id, chunk_index, metrics) when computed so
    the UI can update before slow dependent models (e.g. Ollama) finish.

    Returns:
        The metrics dict, or None if inputs are insufficient.
    """
    whisper_out = results.get("whisper") or {}
    mediapipe_out = results.get("mediapipe") or {}
    vocaltone_out = results.get("vocaltone") or {}

    def _model_ok(payload: Dict[str, Any]) -> bool:
        return isinstance(payload, dict) and "error" not in payload

    if not (_model_ok(whisper_out) or _model_ok(mediapipe_out) or _model_ok(vocaltone_out)):
        logger.warning(
            "[InterviewMetrics] Skipped %s:%s — whisper/mediapipe/vocaltone unavailable",
            session_id,
            chunk_index,
        )
        return None

    metrics = compute_interview_metrics(
        session_id=session_id,
        chunk_index=chunk_index,
        whisper_data=whisper_out,
        mediapipe_data=mediapipe_out,
        vocaltone_data=vocaltone_out,
    )
    results["interview_metrics"] = metrics

    sr = metrics.get("speaking_rate", {})
    st = metrics.get("stress_level", {})
    logger.info(
        "[InterviewMetrics] chunk %s:%s word_count=%s wpm=%s rate=%s stress=%s%%",
        session_id,
        chunk_index,
        whisper_out.get("word_count"),
        sr.get("wpm"),
        sr.get("classification"),
        st.get("stress_percent"),
    )

    if on_metrics_ready:
        try:
            on_metrics_ready(session_id, chunk_index, metrics)
        except Exception as callback_err:
            logger.warning(
                "[InterviewMetrics] on_metrics_ready callback failed: %s",
                callback_err,
            )

    return metrics


def compute_interview_metrics(
    session_id: str,
    chunk_index: int,
    whisper_data: Dict[str, Any],
    mediapipe_data: Dict[str, Any],
    vocaltone_data: Dict[str, Any],
    chunk_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run both derived metrics and return a combined payload for storage/UI.

    Args:
        session_id: Session identifier
        chunk_index: Chunk index
        whisper_data: Whisper model output
        mediapipe_data: MediaPipe model output
        vocaltone_data: VocalTone model output
        chunk_duration_sec: Override duration; falls back to whisper audio_duration_sec

    Returns:
        Combined metrics dict with speaking_rate and stress_level keys.
    """
    duration = chunk_duration_sec
    if duration is None:
        duration = _safe_float(whisper_data.get("audio_duration_sec"), 30.0)
    if duration is None or duration <= 0:
        duration = _safe_float(mediapipe_data.get("video_duration_sec"), 30.0)
    if duration is None or duration <= 0:
        duration = 30.0

    word_count = int(whisper_data.get("word_count") or 0)
    wpm = (word_count / max(duration, 1e-3)) * 60.0
    pitch_std = _safe_float(vocaltone_data.get("pitch_std"), 0.0) or 0.0
    energy_level = _safe_float(vocaltone_data.get("energy_level"), 0.0) or 0.0
    movement_variance = _safe_float(mediapipe_data.get("movement_variance"), 0.0) or 0.0

    baseline_manager.establish_if_needed(
        session_id,
        chunk_index,
        wpm=wpm,
        pitch_std=pitch_std,
        energy_level=energy_level,
        movement_variance=movement_variance,
    )

    speaking_rate = calculate_speaking_rate(
        whisper_data, duration, session_id, chunk_index
    )
    stress_level = calculate_stress_level(
        vocaltone_data, mediapipe_data, session_id, chunk_index
    )

    baseline = baseline_manager.get(session_id)
    return {
        "speaking_rate": speaking_rate,
        "stress_level": stress_level,
        "baseline": {
            "established": baseline.established,
            "baseline_wpm": baseline.baseline_wpm,
            "baseline_pitch_std": baseline.baseline_pitch_std,
            "baseline_energy_level": baseline.baseline_energy_level,
            "baseline_movement_variance": baseline.baseline_movement_variance,
            "calibration_chunk_index": baseline.calibration_chunk_index,
        },
    }


def extract_pose_vector(pose_landmarks_list: Any) -> Optional[Any]:
    """
    Flatten upper-body pose landmarks into a numeric vector for motion tracking.

    Returns:
        numpy.ndarray shape (N,) or None if pose not detected.
    """
    import numpy as np

    if not pose_landmarks_list or len(pose_landmarks_list) == 0:
        return None

    landmarks = pose_landmarks_list[0]
    key_indices = (0, 11, 12, 13, 14, 15, 16, 23, 24)
    coords: List[float] = []
    for idx in key_indices:
        if idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        z = getattr(lm, "z", 0.0) or 0.0
        coords.extend([float(lm.x), float(lm.y), float(z)])

    if not coords:
        return None
    return np.array(coords, dtype=np.float64)


def compute_pose_movement_metrics(pose_vectors: List[Any]) -> Tuple[float, float]:
    """
    Quantify inter-frame pose change across a video chunk.

    Args:
        pose_vectors: List of pose vectors from consecutive sampled frames

    Returns:
        (movement_variance, posture_delta_mean)
        - movement_variance: variance of frame-to-frame L2 displacements
        - posture_delta: mean frame-to-frame displacement (posture_delta)
    """
    import numpy as np

    if len(pose_vectors) < 2:
        return 0.0, 0.0

    deltas: List[float] = []
    for i in range(1, len(pose_vectors)):
        delta = float(np.linalg.norm(pose_vectors[i] - pose_vectors[i - 1]))
        deltas.append(delta)

    if not deltas:
        return 0.0, 0.0

    movement_variance = float(np.var(deltas))
    posture_delta = float(np.mean(deltas))
    return movement_variance, posture_delta
