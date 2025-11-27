"""
run_dnn.py
-----------
Train and evaluate a PyTorch Deep Neural Network (DNN) for phishing website detection.

This script:
1. Loads pre-split train/test data from data/processed/
2. Builds and trains a PyTorch DNN with architecture search and hyperparameter tuning
3. Tests different network architectures: [64,32], [128,64], [256,128,64], [128,64,32]
4. Evaluates the best model on the test set
5. Saves performance metrics to results/metrics_summary.csv (appends if exists)
6. Generates and saves a confusion matrix visualization

Features:
    - Automatic feature scaling (StandardScaler)
    - Dropout regularization (0.2)
    - Adam optimizer with learning rate 0.001
    - Binary cross-entropy loss
    - Architecture search with F1-score validation

Outputs:
    - PyTorch model state: results/neural_network.pth
    - Full model object: results/neural_network.pkl (includes scaler)
    - Metrics appended to results/metrics_summary.csv
    - Confusion matrix: results/confusion_matrices/neural_network.png
"""

from data_preprocessing import load_split_data
from model_training import build_models, train_model
from evaluate_models import evaluate, save_results

X_train, X_test, y_train, y_test = load_split_data()

add_models = build_models()
dnn_model = add_models["Neural Network"]
metrics_list = []

best_dnn = train_model(dnn_model, X_train, y_train, "Neural Network")

metrics = evaluate(best_dnn, X_train, y_train, X_test, y_test, "Neural Network")
metrics_list.append(metrics)

save_results(metrics_list)
