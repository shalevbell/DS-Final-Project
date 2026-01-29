"""
Train Vocal Tone emotion classification model.

This script:
1. Loads processed dataset (X, y from process_savee_dataset_for_training)
2. Splits into 80% train / 20% test
3. Trains multiple models and selects the best one
4. Evaluates performance
5. Saves the best model + scaler + labels_map
"""

import sys
import os
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from scipy.stats import uniform, randint

# Add parent directory (backend) to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from run_models import process_savee_dataset_for_training


def train_models(X_train, X_val, y_train, y_val, use_hyperparameter_tuning=True):
    """
    Train multiple models with hyperparameter tuning and return their performances.
    
    Args:
        use_hyperparameter_tuning: Whether to use RandomizedSearchCV for tuning (default: True)
    
    Returns:
        List of tuples: (model_name, model, accuracy, f1_score)
    """
    models = {}
    results = []
    
    print("\n" + "=" * 80)
    print("Training Models with Hyperparameter Tuning" if use_hyperparameter_tuning else "Training Models")
    print("=" * 80)
    
    # 1. SVM with RBF kernel (with hyperparameter tuning)
    print("\n1. Training SVM (RBF kernel) with hyperparameter tuning...")
    if use_hyperparameter_tuning:
        svm_param_dist = {
            'C': uniform(0.1, 2.0),  # 0.1 to 2.1
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
        }
        svm_base = SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        svm_search = RandomizedSearchCV(
            svm_base,
            svm_param_dist,
            n_iter=20,  # Try 20 random combinations
            cv=3,  # 3-fold cross-validation
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        svm_search.fit(X_train, y_train)
        svm_model = svm_search.best_estimator_
        print(f"   🔧 Best params: {svm_search.best_params_}")
        print(f"   📊 Best CV score: {svm_search.best_score_:.4f}")
    else:
        svm_model = SVC(
            kernel='rbf',
            C=0.5,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        svm_model.fit(X_train, y_train)
    
    svm_pred = svm_model.predict(X_val)
    svm_accuracy = accuracy_score(y_val, svm_pred)
    svm_f1 = f1_score(y_val, svm_pred, average='weighted')
    models['SVM_RBF'] = svm_model
    results.append(('SVM_RBF', svm_model, svm_accuracy, svm_f1))
    print(f"   ✅ SVM Accuracy: {svm_accuracy:.4f}, F1: {svm_f1:.4f}")
    
    # 2. SVM with Linear kernel (with hyperparameter tuning)
    print("\n2. Training SVM (Linear kernel) with hyperparameter tuning...")
    if use_hyperparameter_tuning:
        svm_linear_param_dist = {
            'C': uniform(0.1, 2.0)  # 0.1 to 2.1
        }
        svm_linear_base = SVC(
            kernel='linear',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        svm_linear_search = RandomizedSearchCV(
            svm_linear_base,
            svm_linear_param_dist,
            n_iter=15,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        svm_linear_search.fit(X_train, y_train)
        svm_linear = svm_linear_search.best_estimator_
        print(f"   🔧 Best params: {svm_linear_search.best_params_}")
        print(f"   📊 Best CV score: {svm_linear_search.best_score_:.4f}")
    else:
        svm_linear = SVC(
            kernel='linear',
            C=0.5,
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        svm_linear.fit(X_train, y_train)
    
    svm_linear_pred = svm_linear.predict(X_val)
    svm_linear_accuracy = accuracy_score(y_val, svm_linear_pred)
    svm_linear_f1 = f1_score(y_val, svm_linear_pred, average='weighted')
    models['SVM_Linear'] = svm_linear
    results.append(('SVM_Linear', svm_linear, svm_linear_accuracy, svm_linear_f1))
    print(f"   ✅ SVM Linear Accuracy: {svm_linear_accuracy:.4f}, F1: {svm_linear_f1:.4f}")
    
    # 3. Random Forest (with hyperparameter tuning)
    print("\n3. Training Random Forest with hyperparameter tuning...")
    if use_hyperparameter_tuning:
        rf_param_dist = {
            'n_estimators': randint(100, 300),  # 100 to 299
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_base = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_search = RandomizedSearchCV(
            rf_base,
            rf_param_dist,
            n_iter=20,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        rf_search.fit(X_train, y_train)
        rf_model = rf_search.best_estimator_
        print(f"   🔧 Best params: {rf_search.best_params_}")
        print(f"   📊 Best CV score: {rf_search.best_score_:.4f}")
    else:
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
    
    rf_pred = rf_model.predict(X_val)
    rf_accuracy = accuracy_score(y_val, rf_pred)
    rf_f1 = f1_score(y_val, rf_pred, average='weighted')
    models['RandomForest'] = rf_model
    results.append(('RandomForest', rf_model, rf_accuracy, rf_f1))
    print(f"   ✅ Random Forest Accuracy: {rf_accuracy:.4f}, F1: {rf_f1:.4f}")
    
    # 4. Neural Network (MLP) (with hyperparameter tuning)
    print("\n4. Training Neural Network (MLP) with hyperparameter tuning...")
    if use_hyperparameter_tuning:
        mlp_param_dist = {
            'hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64), (512, 256)],
            'alpha': uniform(0.0001, 0.1),  # 0.0001 to 0.1001
            'learning_rate_init': uniform(0.0001, 0.01)  # 0.0001 to 0.0101
        }
        mlp_base = MLPClassifier(
            max_iter=1000,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        mlp_search = RandomizedSearchCV(
            mlp_base,
            mlp_param_dist,
            n_iter=15,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        mlp_search.fit(X_train, y_train)
        mlp_model = mlp_search.best_estimator_
        print(f"   🔧 Best params: {mlp_search.best_params_}")
        print(f"   📊 Best CV score: {mlp_search.best_score_:.4f}")
    else:
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=1000,
            alpha=0.01,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        mlp_model.fit(X_train, y_train)
    
    mlp_pred = mlp_model.predict(X_val)
    mlp_accuracy = accuracy_score(y_val, mlp_pred)
    mlp_f1 = f1_score(y_val, mlp_pred, average='weighted')
    models['MLP'] = mlp_model
    results.append(('MLP', mlp_model, mlp_accuracy, mlp_f1))
    print(f"   ✅ MLP Accuracy: {mlp_accuracy:.4f}, F1: {mlp_f1:.4f}")
    
    return models, results


def evaluate_model(model, X_test, y_test, labels_map, model_name=""):
    """Evaluate model and print detailed metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "=" * 80)
    print(f"Evaluation Results{': ' + model_name if model_name else ''}")
    print("=" * 80)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Weighted F1 Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[labels_map[i] for i in sorted(labels_map.keys())]
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, label_name in sorted(labels_map.items()):
        class_indices = y_test == i
        if class_indices.sum() > 0:
            class_accuracy = (y_pred[class_indices] == i).sum() / class_indices.sum()
            print(f"  {label_name:15s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm
    }


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("Vocal Tone Emotion Classification Model Training")
    print("=" * 80)
    print()
    
    # Step 1: Load and process dataset
    print("Step 1: Loading and processing dataset...")
    print("   Using extended features: MFCC + Chroma + Spectral (107 features)")
    print("   Using data augmentation: Yes (time_stretch, pitch_shift, noise, time_shift)")
    try:
        X, y, labels_map = process_savee_dataset_for_training(
            dataset_path=None,
            target_sr=16000,
            target_duration_sec=3.0,
            n_mfcc=40,
            show_progress=True,
            use_augmentation=True,
            augmentation_factor=2  # Creates 2 augmented versions per original file
        )
        print(f"\n✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"✅ Classes: {len(labels_map)} - {list(labels_map.values())}")
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Split into train (80%) and test (20%)
    print("\n" + "=" * 80)
    print("Step 2: Splitting dataset (80% train, 20% test)")
    print("=" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Maintain class distribution
    )
    print(f"✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    # Step 3: Feature scaling
    print("\n" + "=" * 80)
    print("Step 3: Scaling features")
    print("=" * 80)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled (StandardScaler)")
    
    # Step 4: Further split train into train/validation for model selection
    print("\n" + "=" * 80)
    print("Step 4: Creating validation set from training data")
    print("=" * 80)
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train,
        test_size=0.2,  # 20% of training data for validation
        random_state=42,
        stratify=y_train
    )
    print(f"✅ Final train set: {X_train_final.shape[0]} samples")
    print(f"✅ Validation set: {X_val.shape[0]} samples")
    print(f"✅ Test set (reserved): {X_test_scaled.shape[0]} samples")
    
    # Step 5: Train multiple models and compare
    _, results = train_models(X_train_final, X_val, y_train_final, y_val)
    
    # Step 6: Select best model based on F1 score
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print("\nValidation Set Performance:")
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12}")
    print("-" * 44)
    for name, _, acc, f1 in results:
        print(f"{name:<20} {acc:.4f}       {f1:.4f}")
    
    # Select best model (highest F1 score)
    best_result = max(results, key=lambda x: x[3])  # x[3] is F1 score
    best_model_name, _, best_val_acc, best_val_f1 = best_result
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Validation Accuracy: {best_val_acc:.4f}")
    print(f"   Validation F1 Score: {best_val_f1:.4f}")
    
    # Step 7: Retrain best model on full training set (train + validation)
    print("\n" + "=" * 80)
    print("Step 7: Retraining best model on full training set")
    print("=" * 80)
    print(f"Retraining {best_model_name} on {X_train_scaled.shape[0]} samples...")
    
    # Retrain on full training set (with same improved hyperparameters)
    if best_model_name == 'SVM_RBF':
        final_model = SVC(
            kernel='rbf',
            C=0.5,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    elif best_model_name == 'SVM_Linear':
        final_model = SVC(
            kernel='linear',
            C=0.5,
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    elif best_model_name == 'RandomForest':
        final_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif best_model_name == 'MLP':
        final_model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=1000,
            alpha=0.01,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
    
    final_model.fit(X_train_scaled, y_train)
    print("✅ Model retrained on full training set")
    
    # Step 8: Final evaluation on test set
    print("\n" + "=" * 80)
    print("Step 8: Final Evaluation on Test Set")
    print("=" * 80)
    test_results = evaluate_model(
        final_model, X_test_scaled, y_test, labels_map, best_model_name
    )
    
    # Step 9: Save model and artifacts
    print("\n" + "=" * 80)
    print("Step 9: Saving Model")
    print("=" * 80)
    
    # Create models directory if it doesn't exist
    models_dir = backend_dir / 'models' / 'vocal_tone'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / 'vocal_tone_model.pkl'
    scaler_path = models_dir / 'vocal_tone_scaler.pkl'
    labels_path = models_dir / 'vocal_tone_labels.pkl'
    
    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(labels_map, labels_path)
    
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Scaler saved: {scaler_path}")
    print(f"✅ Labels map saved: {labels_path}")
    
    # Step 10: Summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Set Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"Test Set F1 Score: {test_results['f1_score']:.4f}")
    print(f"\nModel files saved in: {models_dir}")
    print("\nTo use the model:")
    print("  model = joblib.load('models/vocal_tone/vocal_tone_model.pkl')")
    print("  scaler = joblib.load('models/vocal_tone/vocal_tone_scaler.pkl')")
    print("  labels = joblib.load('models/vocal_tone/vocal_tone_labels.pkl')")
    
    return final_model, scaler, labels_map, test_results


if __name__ == '__main__':
    result = main()
    if result is None:
        raise SystemExit(1)
    model, scaler, labels, results = result

