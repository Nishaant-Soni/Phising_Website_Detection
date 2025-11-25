"""
evaluate_models.py
------------------
Evaluates model performance with classification metrics, confusion matrix, ROC-AUC.
Saves results to CSV and plots.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate(model, X_train, y_train, X_test, y_test, name):
    """Compute metrics for a given model."""
    # Train predictions and accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Test predictions and metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model": name,
        "Train_Accuracy": train_accuracy,
        "Test_Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }
    
    # Create output directories if they don't exist
    os.makedirs("results/confusion_matrices", exist_ok=True)
    os.makedirs("results/roc_curves", exist_ok=True)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"results/confusion_matrices/{name.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {metrics["ROC-AUC"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve: {name}', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(f"results/roc_curves/{name.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics

def save_results(metrics_list):
    """Save all model metrics to a summary CSV."""
    df = pd.DataFrame(metrics_list)
    csv_path = "results/metrics_summary.csv"
    file_exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
