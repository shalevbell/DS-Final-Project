"""
Clifton Strengths domain classifier using Whisper + VocalTone fusion.
"""
import os
import pickle
import logging
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class CliftonFusionModel:
    """Predicts Clifton Strengths domain from multi-modal features."""

    DOMAIN_KEYWORDS = {
        'Executing': ['complete', 'achieve', 'finish', 'deliver', 'organize', 'execute', 'done', 'task',
                      'deadline', 'results', 'accomplish', 'implement', 'action'],
        'Influencing': ['convince', 'inspire', 'lead', 'present', 'motivate', 'impact', 'persuade',
                        'influence', 'sell', 'command', 'competition', 'win'],
        'Relationship': ['connect', 'understand', 'support', 'help', 'team', 'collaborate', 'together',
                         'empathy', 'harmony', 'relationship', 'care', 'include'],
        'Strategic': ['future', 'plan', 'vision', 'analyze', 'think', 'strategy', 'goal', 'idea',
                      'concept', 'pattern', 'possibility', 'context', 'perspective']
    }

    EMOTION_MAPPING = {
        'happy': 1.0,
        'engaged': 0.8,
        'focused': 0.7,
        'neutral': 0.5,
        'sad': 0.2,
        'angry': 0.0
    }

    def __init__(self, model_dir: str, development_threshold: Optional[float] = None):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None

        # Use config value if not explicitly provided
        if development_threshold is None:
            from config import Config
            development_threshold = Config.CLIFTON_DEVELOPMENT_THRESHOLD

        self.development_threshold = development_threshold

    def load_model(self):
        """Lazy load model from disk."""
        model_path = os.path.join(self.model_dir, 'clifton_fusion_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'clifton_scaler.pkl')
        labels_path = os.path.join(self.model_dir, 'clifton_labels.pkl')

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(labels_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        logger.info('[CliftonFusion] Model loaded successfully')

    def extract_features(
        self,
        whisper_result: Dict,
        vocaltone_result: Dict
    ) -> np.ndarray:
        """Extract features from model outputs."""
        features = []

        # Transcript features
        transcript = whisper_result.get('transcript', '')
        words = transcript.lower().split()
        word_count = len(words)

        features.append(word_count)  # Word count
        # Assume 30-second chunks for speech rate calculation
        features.append(word_count / 30.0 if word_count > 0 else 0)  # Words per second
        features.append(whisper_result.get('confidence', 0.0))

        # Keyword matching scores for each domain
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for word in words if word in keywords) / max(word_count, 1)
            features.append(score)

        # Vocal features
        features.append(vocaltone_result.get('pitch_mean', 0.0))
        features.append(vocaltone_result.get('pitch_std', 0.0))
        features.append(vocaltone_result.get('tempo', 0.0))
        features.append(vocaltone_result.get('energy_level', 0.0))

        # Emotion mapping
        emotion = vocaltone_result.get('emotion', 'neutral')
        emotion_score = self.EMOTION_MAPPING.get(emotion, 0.5)
        features.append(emotion_score)
        features.append(vocaltone_result.get('confidence', 0.0))

        return np.array(features).reshape(1, -1)

    def predict(
        self,
        whisper_result: Dict,
        vocaltone_result: Dict
    ) -> Dict:
        """Predict Clifton domain."""
        if self.model is None:
            self.load_model()

        # Extract and scale features
        features = self.extract_features(whisper_result, vocaltone_result)
        features_scaled = self.scaler.transform(features)

        # Predict
        predicted_idx = self.model.predict(features_scaled)[0]
        predicted_domain = self.label_encoder.inverse_transform([predicted_idx])[0]

        # Get probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        domain_probs = {
            self.label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(probabilities)
        }

        confidence = probabilities[predicted_idx]

        # Identify development opportunities (domains with low probabilities)
        development_opportunities = [
            domain for domain, prob in domain_probs.items()
            if prob < self.development_threshold
        ]

        # Sort by probability (lowest first) for prioritization
        development_opportunities.sort(key=lambda d: domain_probs[d])

        return {
            'predicted_domain': predicted_domain,
            'confidence': float(confidence),
            'domain_probabilities': domain_probs,
            'development_opportunities': development_opportunities
        }
