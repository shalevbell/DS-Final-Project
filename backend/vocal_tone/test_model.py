"""
Test script to verify the trained vocal tone model works correctly.

This script loads the saved model and shows what's inside.
"""

import sys
from pathlib import Path
import joblib
import numpy as np

# Add parent directory (backend) to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Path to model files
models_dir = backend_dir / 'models' / 'vocal_tone'
model_path = models_dir / 'vocal_tone_model.pkl'
scaler_path = models_dir / 'vocal_tone_scaler.pkl'
labels_path = models_dir / 'vocal_tone_labels.pkl'


def main():
    """Load and inspect the trained model."""
    print("=" * 80)
    print("Testing Trained Vocal Tone Model")
    print("=" * 80)
    print()
    
    # Check if files exist
    print("Step 1: Checking model files...")
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    if not scaler_path.exists():
        print(f"❌ Scaler file not found: {scaler_path}")
        return
    if not labels_path.exists():
        print(f"❌ Labels file not found: {labels_path}")
        return
    
    print("✅ All model files exist!")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Labels: {labels_path}")
    print()
    
    # Load files
    print("Step 2: Loading model files...")
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        labels_map = joblib.load(labels_path)
        print("✅ All files loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Inspect model
    print("Step 3: Model Information")
    print("=" * 80)
    print(f"Model type: {type(model).__name__}")
    print(f"Model class: {model.__class__.__module__}.{model.__class__.__name__}")
    
    # Show model parameters if available
    if hasattr(model, 'get_params'):
        print("\nModel parameters:")
        params = model.get_params()
        for key, value in list(params.items())[:5]:  # Show first 5
            print(f"  {key}: {value}")
        if len(params) > 5:
            print(f"  ... and {len(params) - 5} more parameters")
    
    print()
    
    # Inspect scaler
    print("Step 4: Scaler Information")
    print("=" * 80)
    print(f"Scaler type: {type(scaler).__name__}")
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        print(f"Features scaled: {len(scaler.mean_)}")
        print(f"Mean range: [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}]")
        print(f"Scale range: [{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]")
    
    print()
    
    # Inspect labels
    print("Step 5: Labels Map")
    print("=" * 80)
    print(f"Number of classes: {len(labels_map)}")
    print("Classes:")
    for label_idx in sorted(labels_map.keys()):
        label_name = labels_map[label_idx]
        print(f"  {label_idx}: {label_name}")
    
    print()
    
    # Test prediction with dummy data
    print("Step 6: Testing Model Prediction")
    print("=" * 80)
    print("Creating dummy feature vector (80 features, like real audio)...")
    
    # Create dummy features (simulating processed audio)
    dummy_features = np.random.randn(80)  # Random features
    print(f"Input shape: {dummy_features.shape}")
    
    # Scale features
    dummy_features_scaled = scaler.transform([dummy_features])
    print("✅ Features scaled")
    
    # Predict
    prediction = model.predict(dummy_features_scaled)[0]
    probabilities = model.predict_proba(dummy_features_scaled)[0]
    
    predicted_emotion = labels_map[prediction]
    print(f"✅ Prediction: {predicted_emotion} (class {prediction})")
    
    print("\nProbability distribution:")
    for i, prob in enumerate(probabilities):
        emotion = labels_map[i]
        print(f"  {emotion:15s}: {prob:.4f} ({prob*100:.2f}%)")
    
    print()
    print("=" * 80)
    print("✅ Model is ready to use!")
    print("=" * 80)
    print()
    print("To use the model in your code:")
    print("""
import joblib
from pathlib import Path

# Load
model = joblib.load('models/vocal_tone/vocal_tone_model.pkl')
scaler = joblib.load('models/vocal_tone/vocal_tone_scaler.pkl')
labels = joblib.load('models/vocal_tone/vocal_tone_labels.pkl')

# Process audio → get 80 features (like in process_savee_dataset_for_training)
features = extract_features_from_audio(audio_bytes)  # returns 80 features

# Scale
features_scaled = scaler.transform([features])

# Predict
emotion_idx = model.predict(features_scaled)[0]
emotion_name = labels[emotion_idx]

print(f"Predicted emotion: {emotion_name}")
""")


if __name__ == '__main__':
    main()

