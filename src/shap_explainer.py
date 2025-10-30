"""
shap_explainer.py
-----------------
Uses SHAP to interpret model predictions and visualize feature importance.
"""

import shap
import matplotlib.pyplot as plt

def shap_analysis(model, X_train):
    """Run SHAP summary plot for feature importance."""
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig("results/shap_plots/shap_summary.png", bbox_inches='tight')
    plt.close()
