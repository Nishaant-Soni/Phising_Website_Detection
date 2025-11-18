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
from sklearn.model_selection import GridSearchCV
import joblib


def build_models():
    """Initialize multiple models for comparison."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",      # good for small/medium data, L2 penalty
            random_state=42,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            probability=True,        # needed for predict_proba in evaluation
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        ),
    }

    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting="soft"
    )
    models["Ensemble"] = ensemble
    return models


def train_model(model, X_train, y_train, model_name):
    """
    Train (with hyperparameter tuning for LR and SVM) and save the model.
    Returns the best model.
    """
    best_model = model

    # --- Logistic Regression tuning ---
    if model_name == "Logistic Regression":
        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],   # if you want L1, switch solver to 'saga'
        }
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        print("Best params for Logistic Regression:", grid.best_params_)
        best_model = grid.best_estimator_

    # --- SVM (RBF) tuning ---
    elif model_name == "SVM (RBF)":
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", 0.01, 0.1, 1],
        }
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        print("Best params for SVM (RBF):", grid.best_params_)
        best_model = grid.best_estimator_

    else:
        # Other models: just fit directly (RF, NN, Ensemble)
        best_model.fit(X_train, y_train)

    # Save and return
    joblib.dump(best_model, f"results/{model_name.replace(' ', '_').lower()}.pkl")
    return best_model

