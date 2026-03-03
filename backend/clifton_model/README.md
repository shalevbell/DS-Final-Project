# Clifton Strengths Fusion Model

This directory contains the Clifton Strengths domain classifier that predicts one of 4 domains (Executing, Influencing, Relationship, Strategic) from Whisper transcription and VocalTone analysis.

## Quick Start

### 1. Train the Model

```bash
cd backend/clifton_model
python train_fusion.py
```

This will:
- Load training data from `training_data/data.json` (196 samples)
- Extract 13 features per sample (transcript + vocal + keyword matching)
- Train a Random Forest classifier with 80/20 train/test split
- Save model artifacts to `models/` directory:
  - `clifton_fusion_model.pkl` - Random Forest model
  - `clifton_scaler.pkl` - StandardScaler for feature normalization
  - `clifton_labels.pkl` - LabelEncoder for domain classes

### 2. Upload Model to Google Drive (for Docker deployment)

```bash
cd backend/clifton_model/models
zip ../clifton_model.zip clifton_fusion_model.pkl clifton_scaler.pkl clifton_labels.pkl
```

Then:
1. Upload `clifton_model.zip` to Google Drive
2. Share with "Anyone with the link can view"
3. Copy the file ID from the share link (format: `https://drive.google.com/file/d/FILE_ID/view`)
4. Add to `docker-compose.yml`: `DRIVE_CLIFTON_MODEL_ZIP_ID: "YOUR_FILE_ID"`

## Architecture

### Feature Engineering (13 features)

**From Whisper transcription:**
- Word count
- Speech rate (words per second)
- Transcription confidence

**Domain keyword matching (4 features):**
- Executing keywords: complete, achieve, finish, deliver, organize, execute, done, task, deadline, results
- Influencing keywords: convince, inspire, lead, present, motivate, impact, persuade, influence, sell
- Relationship keywords: connect, understand, support, help, team, collaborate, together, empathy, harmony
- Strategic keywords: future, plan, vision, analyze, think, strategy, goal, idea, concept, pattern

**From VocalTone analysis:**
- Pitch mean (Hz)
- Pitch standard deviation
- Tempo (BPM)
- Energy level (0-1)
- Emotion score (mapped from emotion label to 0-1)
- Emotion confidence

### Model

- **Algorithm**: Random Forest (100 trees, max_depth=10)
- **Why Random Forest?**
  - Handles small datasets well
  - Built-in feature importance
  - No hyperparameter sensitivity
  - Fast inference (~10-50ms)
  - Interpretable results

### Integration

The model integrates with the existing ML pipeline through:
- **Sequential execution**: Runs AFTER Whisper and VocalTone complete
- **Dependency injection**: Receives Whisper/VocalTone outputs as input
- **Real-time streaming**: Streams predictions to frontend via WebSocket
- **Redis storage**: Results stored in `chunks:{sessionId}:{chunkIndex}:results:clifton_fusion`

## Files

- `fusion_model.py` - CliftonFusionModel class (feature extraction + prediction)
- `train_fusion.py` - Training script
- `training_data/data.json` - LLM-generated training data (196 samples)
- `models/` - Trained model artifacts (created after training)

## Training Data Format

```json
{
  "domain": "Executing",
  "transcript": "I made sure to complete all tasks on time...",
  "pitch_mean": 175.0,
  "pitch_std": 22.0,
  "tempo": 95.0,
  "energy_level": 0.6,
  "emotion": "neutral",
  "transcript_confidence": 0.92,
  "emotion_confidence": 0.88
}
```

## Testing

After training, test the model:

```python
from fusion_model import CliftonFusionModel

model = CliftonFusionModel('models')
model.load_model()

# Test prediction
whisper_result = {
    'transcript': 'I want to achieve my goals and complete the task',
    'confidence': 0.95
}
vocaltone_result = {
    'pitch_mean': 180.0,
    'pitch_std': 25.0,
    'tempo': 95.0,
    'energy_level': 0.65,
    'emotion': 'neutral',
    'confidence': 0.9
}

result = model.predict(whisper_result, vocaltone_result)
print(f"Predicted domain: {result['predicted_domain']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All probabilities: {result['domain_probabilities']}")
```
