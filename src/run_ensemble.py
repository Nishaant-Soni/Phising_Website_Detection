"""
run_ensemble.py
---------------
Train and evaluate an ensemble model combining top-performing classifiers.

This script:
1. Loads pre-split train/test data from data/processed/
2. Loads pre-trained individual models (RF, SVM, DNN)
3. Creates a Voting Classifier ensemble with soft voting
4. Evaluates the ensemble on test set
5. Saves performance metrics

Ensemble Strategy:
    - Voting Classifier with soft voting (weighted by predicted probabilities)
    - Combines: Random Forest + SVM (RBF) + Neural Network
    - Uses the strengths of different model types for robust predictions

Prerequisites:
    - Run individual model training scripts first:
      * python src/run_rf.py
      * python src/run_svm.py
      * python src/run_dnn.py

Outputs:
    - Trained ensemble saved: results/ensemble.pkl
    - Metrics appended to results/metrics_summary.csv
    - Confusion matrix: results/confusion_matrices/ensemble.png
"""

import joblib
import os
import numpy as np
from data_preprocessing import load_split_data
from evaluate_models import evaluate, save_results
from model_training import build_ensemble


class SimpleEnsemble:
    """Simple ensemble wrapper that averages predictions from multiple models."""
    
    def __init__(self, models):
        self.models = models
        self.classes_ = np.array([-1, 1])
    
    def _get_positive_class_probs(self, X):
        """Get probabilities for positive class (1) from all models."""
        probs = []
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                probs_all = model.predict_proba(X)
                if hasattr(model, 'classes_'):
                    try:
                        pos_idx = list(model.classes_).index(1)
                    except ValueError:
                        pos_idx = 1  # fallback
                else:
                    pos_idx = 1
                probs.append(probs_all[:, pos_idx])
        return np.mean(probs, axis=0) if probs else np.zeros(len(X))
    
    def predict(self, X):
        ensemble_probs = self._get_positive_class_probs(X)
        return np.where(ensemble_probs > 0.5, 1, -1)
    
    def predict_proba(self, X):
        ensemble_probs = self._get_positive_class_probs(X)
        return np.column_stack([1 - ensemble_probs, ensemble_probs])


def load_pretrained_models():
    """Load the best-performing pre-trained models for ensemble."""
    models = {}
    model_configs = [
        ('Random Forest', 'results/random_forest.pkl'),
        ('SVM (RBF)', 'results/svm_(rbf).pkl'),
        ('Neural Network', 'results/neural_network.pkl')
    ]
    
    for name, path in model_configs:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                models[name] = model
                print(f"Loaded {name}")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
        else:
            print(f"Model not found: {path}")
    
    return models

def main():
    print("Loading training data...")
    X_train, X_test, y_train, y_test = load_split_data()
    
    print("\nLoading pre-trained models for ensemble...")
    trained_models = load_pretrained_models()
    
    if len(trained_models) < 2:
        print("Need at least 2 models for ensemble. Please train more models first.")
        return
    
    print(f"\n Creating ensemble with {len(trained_models)} models")
    print("Using soft voting (averaging predicted probabilities)")
    
    print("\nEvaluating individual models and computing ensemble predictions...")
    
    ensemble = SimpleEnsemble(trained_models)
    
    print("Ensemble created")
    
    print("\nEvaluating ensemble...")
    metrics = evaluate(ensemble, X_train, y_train, X_test, y_test, "Ensemble")
    
    print(f"\nEnsemble Performance:")
    print(f"  Train Accuracy: {metrics['Train_Accuracy']:.4f}")
    print(f"  Test Accuracy:  {metrics['Test_Accuracy']:.4f}")
    print(f"  Precision:      {metrics['Precision']:.4f}")
    print(f"  Recall:         {metrics['Recall']:.4f}")
    print(f"  F1-Score:       {metrics['F1']:.4f}")
    print(f"  ROC-AUC:        {metrics['ROC-AUC']:.4f}")
    
    save_results([metrics])
    
    joblib.dump(ensemble, "results/ensemble.pkl")
    print("\nEnsemble model saved to results/ensemble.pkl")
    print("Metrics appended to results/metrics_summary.csv")

if __name__ == "__main__":
    main()
