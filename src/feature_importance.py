"""
Feature Importance Analysis for All 5 Phishing Detection Models

Unified analysis using permutation importance for fair comparison across all models:
- All models: Permutation importance (model-agnostic approach)
- Evaluates feature impact on model performance using F1-score
- Measures decrease in performance when feature values are shuffled

All analyses performed on test data for realistic generalization assessment.
Generates top 10 feature rankings and clean visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import sys
from data_preprocessing import load_split_data
from sklearn.inspection import permutation_importance

try:
    sys.path.append('src')
    from run_ensemble import SimpleEnsemble
except ImportError:
    print("Warning: Could not import SimpleEnsemble class")
    SimpleEnsemble = None

warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    """Feature importance analyzer for all 5 trained models using appropriate methods for each model type."""
    
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = load_split_data()
        
        # All 5 trained models
        self.top_models = {
            "Neural Network": "results/neural_network.pkl",
            "Random Forest": "results/random_forest.pkl",
            "Ensemble": "results/ensemble.pkl",
            "Logistic Regression": "results/logistic_regression.pkl",
            "SVM (RBF)": "results/svm_(rbf).pkl"
        }
        
        self.feature_names = self.X_train.columns.tolist()
        
    def load_model(self, model_name, model_path):
        """Load model with proper error handling."""
        try:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                print(f"Successfully loaded {model_name}")
                return model
            else:
                print(f"Model file not found: {model_path}")
                return None
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None
    
    def analyze_permutation_importance(self, model, model_name):
        """Use permutation importance for general feature importance."""
        print(f"\n=== Permutation Importance Analysis for {model_name} ===")
        print("Using permutation importance on test data for generalization assessment")
        
        print(f"Computing permutation importance on {len(self.X_test)} test samples...")
        
        if model_name == "Ensemble" and not hasattr(model, 'fit'):
            from sklearn.base import BaseEstimator, ClassifierMixin
            
            class EnsembleWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, ensemble_model):
                    self.model = ensemble_model
                    self.classes_ = ensemble_model.classes_
                
                def fit(self, X, y):
                    return self
                
                def predict(self, X):
                    return self.model.predict(X)
                
                def predict_proba(self, X):
                    return self.model.predict_proba(X)
            
            model = EnsembleWrapper(model)
            print("Created sklearn-compatible wrapper for ensemble")
        
        perm_importance = permutation_importance(
            model, self.X_test, self.y_test, 
            n_repeats=10,  
            random_state=42,
            n_jobs=-1,  
            scoring='f1' 
        )
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)
        
        top_10_features = feature_importance.head(10)
        
        plt.figure(figsize=(15, 10))  
        ax = sns.barplot(x='importance', y='feature', data=top_10_features, palette='plasma')
        
        plt.xlabel('Permutation Importance Score (F1-based)', fontsize=13)
        plt.ylabel('Features', fontsize=13)
        plt.title(f'{model_name}: Top 10 features', 
                 fontsize=15, fontweight='bold', pad=25)
        
        ax.tick_params(axis='y', labelsize=11)
        ax.tick_params(axis='x', labelsize=11)
        plt.subplots_adjust(left=0.25) 
        
        for i, (_, row) in enumerate(top_10_features.iterrows()):
            plt.text(row['importance'] + row['importance']*0.05, i, 
                    f'{row["importance"]:.4f}±{row["std"]:.3f}', 
                    va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout(pad=2.0) 
        
        plot_path = f"results/feature_importance/{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Permutation importance plot saved: {plot_path}")
        
        print(f"\nTop 10 features for general classification using {model_name} permutation importance:")
        for i, (_, row) in enumerate(top_10_features.iterrows(), 1):
            print(f"{i:2}. {row['feature']:<30} : {row['importance']:.4f} ± {row['std']:.3f}")
        
        return top_10_features
    
    def run_comprehensive_analysis(self):
        """Run feature importance analysis for all 5 trained models."""
        print("Starting Feature Importance Analysis for All 5 Trained Models")
        print("- Unified methodology for fair comparison across all architectures")
        print("- Evaluates feature impact on generalization performance")
        print("=" * 80)
        
        results = {}
        
        for model_name, model_path in self.top_models.items():
            print(f"\n{'='*20} {model_name} {'='*20}")
            
            model = self.load_model(model_name, model_path)
            if model is None:
                continue
            
            top_features = self.analyze_permutation_importance(model, model_name)
            results[model_name] = {
                'method': 'permutation_importance',
                'top_features': top_features
            }
            
        print(f"\n{'='*90}")
        print("Feature Importance Analysis Complete for All 5 Models!")
        print(f"Results saved in: results/feature_importance/")
        
        return results

def run_feature_importance_analysis():
    """Main function to run feature importance analysis."""
    os.makedirs("results/feature_importance", exist_ok=True)
    
    analyzer = FeatureImportanceAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    return results

if __name__ == "__main__":
    results = run_feature_importance_analysis()