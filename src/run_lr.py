"""run_lr.py
-----------
Train and evaluate a Logistic Regression model for phishing website detection.

This script:
1. Loads pre-split train/test data from data/processed/
2. Builds and trains a Logistic Regression classifier with hyperparameter tuning
3. Evaluates the model on the test set
4. Saves performance metrics to results/metrics_summary.csv (appends if exists)
5. Generates and saves a confusion matrix visualization

Outputs:
    - Trained model saved in models/ directory
    - Metrics appended to results/metrics_summary.csv
    - Confusion matrix saved to results/confusion_matrices/logistic_regression.png
"""

from data_preprocessing import load_split_data
from model_training import build_models, train_model
from evaluate_models import evaluate, save_results

# Load pre-split data (run prepare_data.py first if not done)
X_train, X_test, y_train, y_test = load_split_data()

all_models = build_models()
metrics_list = []

model = all_models["Logistic Regression"]
best_model = train_model(model, X_train, y_train, "Logistic Regression")

metrics = evaluate(best_model, X_test, y_test, "Logistic Regression")
metrics_list.append(metrics)

save_results(metrics_list)
