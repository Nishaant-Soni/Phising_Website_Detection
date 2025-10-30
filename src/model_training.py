"""
model_training.py
-----------------
Implements ML models: Logistic Regression, SVM, Random Forest, DNN, Ensemble.
Performs hyperparameter tuning and saves trained models.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import joblib

def build_models():
    """Initialize multiple models for comparison."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    ensemble = VotingClassifier(
        estimators=[(n, m) for n, m in models.items()],
        voting='soft'
    )
    models["Ensemble"] = ensemble
    return models

def train_model(model, X_train, y_train, model_name):
    """Train and save individual model."""
    model.fit(X_train, y_train)
    joblib.dump(model, f"results/{model_name.replace(' ', '_').lower()}.pkl")
    return model
