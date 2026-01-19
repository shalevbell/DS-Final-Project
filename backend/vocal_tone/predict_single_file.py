"""
Predict emotion from a single WAV audio file using the trained Vocal Tone model.

This script allows you to test the model on any WAV file to see what emotion it predicts.
"""

import sys
from pathlib import Path
import numpy as np
import joblib
import librosa
import argparse

# Add parent directory (backend) to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


def process_single_audio_file(
    audio_path: str,
    target_sr: int = 16000,
    target_duration_sec: float = 3.0,
    n_mfcc: int = 40
) -> np.ndarray:
    """
    Process a single WAV file and extract features (same as training).
    
    Args:
        audio_path: Path to WAV file
        target_sr: Target sample rate (default: 16000 Hz)
        target_duration_sec: Target duration in seconds (default: 3.0)
        n_mfcc: Number of MFCC coefficients (default: 40)
    
    Returns:
        Feature vector of length 80 (40 mean + 40 std)
    """
    audio_path_obj = Path(audio_path)
    
    if not audio_path_obj.exists():
        raise ValueError(f'Audio file not found: {audio_path}')
    
    if not audio_path_obj.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']:
        print(f"⚠️  Warning: File extension is {audio_path_obj.suffix}, but processing anyway...")
    
    print(f"📁 Loading audio file: {audio_path}")
    
    # Load audio (force mono)
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    print(f"   ✅ Loaded: {len(audio)} samples at {sr}Hz ({len(audio)/sr:.2f} seconds)")
    
    # Resample to target sample rate if needed
    if sr != target_sr:
        print(f"   🔄 Resampling from {sr}Hz to {target_sr}Hz...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Light normalization
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95
        print(f"   ✅ Normalized: max={np.abs(audio).max():.3f}")
    
    # Fix length: crop or pad to target_duration_sec
    target_samples = int(target_duration_sec * target_sr)
    current_samples = len(audio)
    
    if current_samples > target_samples:
        # Crop: take first target_samples
        audio = audio[:target_samples]
        print(f"   ✂️  Cropped to {target_samples} samples ({target_duration_sec}s)")
    elif current_samples < target_samples:
        # Pad: zero-padding at the end
        audio = np.pad(audio, (0, target_samples - current_samples), mode='constant')
        print(f"   ➕ Padded to {target_samples} samples ({target_duration_sec}s)")
    else:
        print(f"   ✅ Already correct length: {target_samples} samples ({target_duration_sec}s)")
    
    # Extract MFCC features
    print(f"   🎵 Extracting MFCC features...")
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=target_sr,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    print(f"   ✅ MFCC shape: {mfccs.shape} (40 coefficients × {mfccs.shape[1]} frames)")
    
    # Convert to fixed vector: mean + std for each MFCC coefficient
    mfcc_mean = np.mean(mfccs, axis=1)  # Mean across time for each coefficient
    mfcc_std = np.std(mfccs, axis=1)    # Std across time for each coefficient
    
    # Concatenate mean and std → vector of length 80
    feature_vector = np.concatenate([mfcc_mean, mfcc_std])
    print(f"   ✅ Feature vector created: shape {feature_vector.shape} (40 mean + 40 std)")
    
    return feature_vector


def predict_emotion(audio_path: str):
    """
    Predict emotion from a single audio file.
    
    Args:
        audio_path: Path to WAV audio file
    """
    print("=" * 80)
    print("Vocal Tone Emotion Prediction")
    print("=" * 80)
    print()
    
    # Step 1: Process audio file
    try:
        features = process_single_audio_file(audio_path)
    except Exception as e:
        print(f"\n❌ Error processing audio file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Step 2: Load model, scaler, and labels
    print("Step 2: Loading trained model...")
    models_dir = backend_dir / 'models' / 'vocal_tone'
    model_path = models_dir / 'vocal_tone_model.pkl'
    scaler_path = models_dir / 'vocal_tone_scaler.pkl'
    labels_path = models_dir / 'vocal_tone_labels.pkl'
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("   Please train the model first using: python vocal_tone/train_model.py")
        return
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        labels_map = joblib.load(labels_path)
        print(f"   ✅ Model loaded: {type(model).__name__}")
        print(f"   ✅ Scaler loaded")
        print(f"   ✅ Labels loaded: {len(labels_map)} classes")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Step 3: Scale features
    print("Step 3: Scaling features...")
    features_scaled = scaler.transform([features])
    print("   ✅ Features scaled")
    
    print()
    
    # Step 4: Predict
    print("Step 4: Predicting emotion...")
    prediction_idx = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    predicted_emotion = labels_map[prediction_idx]
    confidence = probabilities[prediction_idx]
    
    print()
    print("=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"\n🎯 Predicted Emotion: {predicted_emotion}")
    print(f"📊 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print()
    
    print("Probability distribution (all emotions):")
    print(f"{'Emotion':<20} {'Probability':<15} {'Bar'}")
    print("-" * 60)
    
    # Sort by probability (highest first)
    emotion_probs = [(labels_map[i], prob) for i, prob in enumerate(probabilities)]
    emotion_probs.sort(key=lambda x: x[1], reverse=True)
    
    for emotion, prob in emotion_probs:
        bar_length = int(prob * 50)  # Scale to 50 characters
        bar = "█" * bar_length
        print(f"{emotion:<20} {prob:.4f} ({prob*100:>6.2f}%) {bar}")
    
    print()
    print("=" * 80)
    print(f"✅ Prediction complete!")
    print("=" * 80)
    
    return predicted_emotion, confidence, probabilities


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Predict emotion from a WAV audio file using trained Vocal Tone model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vocal_tone/predict_single_file.py /path/to/file.wav
  python vocal_tone/predict_single_file.py "C:\\Users\\shira\\audio.wav"
        """
    )
    
    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to WAV audio file to analyze'
    )
    
    args = parser.parse_args()
    
    # Run prediction
    predict_emotion(args.audio_file)


if __name__ == '__main__':
    # If run without arguments, show help
    if len(sys.argv) == 1:
        print("Usage: python vocal_tone/predict_single_file.py <path_to_wav_file>")
        print("\nExample:")
        print("  python vocal_tone/predict_single_file.py test_audio.wav")
        print("\nOr with full path:")
        print('  python vocal_tone/predict_single_file.py "C:\\Users\\shira\\audio.wav"')
        print()
        
        # Interactive mode: ask for file path
        audio_path = input("Enter path to WAV file (or press Enter to exit): ").strip()
        if audio_path:
            predict_emotion(audio_path)
        else:
            print("Exiting...")
    else:
        main()

