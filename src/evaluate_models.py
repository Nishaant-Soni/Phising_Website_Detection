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
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
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
    """
    Save all model metrics to a summary CSV.
    
    This function properly handles CSV appending by:
    1. Creating the file with header if it doesn't exist
    2. Reading existing data and removing duplicates based on Model name
    3. Appending new results and writing the complete updated data
    4. Ensuring proper newlines and no duplicate entries
    """
    df_new = pd.DataFrame(metrics_list)
    csv_path = "results/metrics_summary.csv"
    
    os.makedirs("results", exist_ok=True)
    
    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            
            models_to_add = df_new['Model'].tolist()
            df_existing = df_existing[~df_existing['Model'].isin(models_to_add)]
            
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not read existing CSV ({e}). Creating new file.")
            df_combined = df_new
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_path, mode='w', header=True, index=False, lineterminator='\n')
    
    print(f"\n✓ Metrics saved to {csv_path}")
    print(f"  Total models in summary: {len(df_combined)}")