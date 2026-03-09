# Vocal Tone Emotion Classification Model

This directory contains the Vocal Tone emotion classifier that predicts emotional states from audio features extracted from interview chunks.

## Quick Start

### 1. Train the Model

```bash
cd backend/vocal_tone_model
python train_model.py
```

This will:
- Load audio files from the SAVEE dataset (`/data/dataset`)
- Extract 107 features per audio file (MFCC + Chroma + Spectral)
- Train SVM/RandomForest/MLP models with hyperparameter tuning
- Save the best model to `models/` directory:
  - `vocal_tone_model.pkl` - Trained classifier
  - `vocal_tone_scaler.pkl` - StandardScaler for feature normalization
  - `vocal_tone_labels.pkl` - Label encoder mapping

### 2. Test the Model

```bash
# Test on a single audio file
python predict_single_file.py /data/dataset/03-01-08-01-01-01-16.wav

# Test on multiple files
python test_model.py
```

### 3. Upload Model to Google Drive (for Docker deployment)

```bash
cd backend/vocal_tone_model/models
zip ../vocal_tone_model.zip vocal_tone_model.pkl vocal_tone_scaler.pkl vocal_tone_labels.pkl
```

Then:
1. Upload `vocal_tone_model.zip` to Google Drive
2. Share with "Anyone with the link can view"
3. Copy the file ID from the share link
4. Add to `docker-compose.yml`: `DRIVE_VOCAL_TONE_MODEL_ZIP_ID: "YOUR_FILE_ID"`

## Architecture

### Feature Engineering (107 features)

**MFCC Features (80 features):**
- 40 MFCC coefficients (mean)
- 40 MFCC coefficients (std)

**Chroma Features (24 features):**
- 12 Chroma coefficients (mean)
- 12 Chroma coefficients (std)

**Spectral Features (3 features):**
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff

### Data Augmentation

Training uses 5 augmentation techniques to increase dataset size:
- Time stretching (0.8x, 1.2x speed)
- Pitch shifting (±2 semitones)
- Background noise injection
- Time shifting

### Model Selection

The training script compares three models:
1. **SVM** (Support Vector Machine) - Good for small datasets
2. **Random Forest** - Ensemble method, handles overfitting well
3. **MLP** (Multi-Layer Perceptron) - Neural network for complex patterns

The best model is automatically selected based on cross-validation accuracy.

### Integration

The model integrates with the ML pipeline through:
- **Independent execution**: Runs in parallel with Whisper and MediaPipe
- **Lazy loading**: Model loads on first use to speed up startup
- **Real-time streaming**: Results streamed to frontend via WebSocket
- **Redis storage**: Results stored in `chunks:{sessionId}:{chunkIndex}:results:vocaltone`

## Files

- `train_model.py` - Main training script with hyperparameter tuning
- `test_model.py` - Evaluate trained model on test set
- `predict_single_file.py` - Test prediction on a single audio file
- `list_dataset.py` - Verify SAVEE dataset is mounted correctly
- `process_dataset.py` - Dataset preprocessing utilities
- `models/` - Trained model artifacts (created after training)

## Dataset Format

The SAVEE dataset should contain WAV audio files with emotion labels:
- Format: `{speaker_id}-{emotion_code}-{recording_id}.wav`
- Emotions: angry, disgust, fear, happiness, neutral, sadness, surprise
- Sample rate: 16kHz (resampled during preprocessing)
- Duration: 3 seconds (padded/cropped during preprocessing)

## Output Format

```python
{
    'emotion': str,                    # Predicted emotion label
    'confidence': float,               # Confidence score (0-1)
    'emotion_probabilities': dict,     # Probabilities for all emotions
    'pitch_mean': float,               # Average pitch (Hz)
    'pitch_std': float,                # Pitch variation
    'tempo': float,                    # Speech tempo (BPM)
    'energy_level': float,             # RMS energy
    'processing_time_ms': int          # Processing time
}
```

## Training Tips

1. **Dataset quality matters**: Ensure SAVEE dataset is properly downloaded
2. **Check feature extraction**: Run `list_dataset.py` to verify dataset structure
3. **Monitor overfitting**: Cross-validation scores should be close to test accuracy
4. **Experiment with augmentation**: Adjust augmentation parameters in `train_model.py`
5. **Model selection**: SVM typically performs best for this dataset size

## Troubleshooting

**Problem**: "Dataset not found"
- **Solution**: Verify `/data/dataset` contains SAVEE audio files
- Check `docker-compose.yml` has correct `DRIVE_DATASET_ZIP_ID`

**Problem**: Low accuracy (<60%)
- **Solution**: Check dataset quality, try different model hyperparameters
- Ensure audio files are not corrupted

**Problem**: "Model file not found" at runtime
- **Solution**: Train the model first or set `DRIVE_VOCAL_TONE_MODEL_ZIP_ID` in docker-compose
