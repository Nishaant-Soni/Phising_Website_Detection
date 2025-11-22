"""run_svm.py
-----------
Train and evaluate a Support Vector Machine (SVM) model for phishing website detection.

This script:
1. Loads pre-split train/test data from data/processed/
2. Builds and trains an SVM classifier with RBF kernel and hyperparameter tuning
3. Evaluates the model on the test set
4. Saves performance metrics to results/metrics_summary.csv (appends if exists)
5. Generates and saves a confusion matrix visualization

Outputs:
    - Trained model saved in models/ directory
    - Metrics appended to results/metrics_summary.csv
    - Confusion matrix saved to results/confusion_matrices/svm_(rbf).png
"""

from data_preprocessing import load_split_data
from model_training import build_models, train_model
from evaluate_models import evaluate, save_results

X_train, X_test, y_train, y_test = load_split_data()

all_models = build_models()
metrics_list = []

model = all_models["SVM (RBF)"]
best_model = train_model(model, X_train, y_train, "SVM (RBF)")

metrics = evaluate(best_model, X_train, y_train, X_test, y_test, "SVM (RBF)")
metrics_list.append(metrics)

save_results(metrics_list)
