"""
run_rf.py
---------
Train and evaluate a Random Forest model for phishing website detection.

This script:
1. Loads pre-split train/test data from data/processed/
2. Builds and trains a Random Forest classifier with hyperparameter tuning
3. Evaluates the model on the test set
4. Saves performance metrics to results/metrics_summary.csv (appends if exists)
5. Generates and saves a confusion matrix visualization

Outputs:
    - Trained model saved via joblib in results/ directory
    - Metrics appended to results/metrics_summary.csv
    - Confusion matrix saved to results/confusion_matrices/random_forest.png
"""

from data_preprocessing import load_split_data
from model_training import build_models, train_model
from evaluate_models import evaluate, save_results

X_train, X_test, y_train, y_test = load_split_data()

all_models = build_models()
rf_model = all_models["Random Forest"]
metrics_list = []

best_rf_model = train_model(rf_model, X_train, y_train, "Random Forest")

metrics = evaluate(best_rf_model, X_train, y_train, X_test, y_test, "Random Forest")
metrics_list.append(metrics)

save_results(metrics_list)



