"""
evaluate_models.py
------------------
Evaluates model performance with classification metrics, confusion matrix, ROC-AUC.
Saves results to CSV and plots.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, X_test, y_test, name):
    """Compute metrics for a given model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.savefig(f"results/confusion_matrices/{name.replace(' ', '_').lower()}.png")
    plt.close()
    return metrics

def save_results(metrics_list):
    """Save all model metrics to a summary CSV."""
    df = pd.DataFrame(metrics_list)
    df.to_csv("results/metrics_summary.csv", index=False)
