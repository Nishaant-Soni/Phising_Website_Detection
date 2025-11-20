"""
run_dnn.py
-----------
Train and evaluate a Neural Network (DNN) model for phishing website detection.

This script:
1. Loads pre-split train/test data from data/processed/
2. Builds and trains a DNN (MLPClassifier) with hyperparameter tuning
3. Evaluates the model on the test set
4. Saves performance metrics to results/metrics_summary.csv (appends if exists)
5. Generates and saves a confusion matrix visualization

Outputs:
    - Trained model saved via joblib in results/
    - Metrics appended to results/metrics_summary.csv
    - Confusion matrix saved to results/confusion_matrices/neural_network.png
"""

from data_preprocessing import load_split_data
from model_training import build_models, train_model
from evaluate_models import evaluate, save_results

X_train, X_test, y_train, y_test = load_split_data()

add_models = build_models()
dnn_model = add_models["Neural Network"]
metrics_list = []


best_dnn = train_model(dnn_model, X_train, y_train, "Neural Network")

metrics = evaluate(best_dnn, X_test, y_test, "Neural Network")
metrics_list.append(metrics)

save_results(metrics_list)
