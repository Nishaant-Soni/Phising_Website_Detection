# Phishing Website Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Status](https://img.shields.io/badge/Status-Active-success)

## Overview

This project implements and compares multiple machine learning models to detect phishing websites based on 30 URL, domain, and content-based features. The models are trained on the **Phishing Website Dataset** from the UCI Machine Learning Repository with 11,055 instances.

**Models Implemented:**
- Logistic Regression (Baseline)
- Support Vector Machine (RBF Kernel)
- Random Forest Classifier
- Deep Neural Network (PyTorch)
- Ensemble Model (Voting Classifier)

**Key Features:**
- SMOTE-based class balancing for improved minority class detection
- Comprehensive hyperparameter tuning with grid search
- Feature importance analysis using permutation importance
- Model evaluation with metrics, confusion matrices, and ROC curves
- Reproducible train/test splits for fair model comparison

---

## Team Information

**Team Members:**  
- Radhika Khurana  
- Nishaant Sitendra Soni  
- Sahil Subodh Bane  

**Professor:** Ehsan Elhamifar  
**Course:** CS6140 — Machine Learning  
**Submission Date:** December 13, 2025

---

## Project Structure

```
Phising_Website_Detection/
├── data/
│   ├── raw/                      # Original dataset files
│   │   ├── Dataset.arff          # Original ARFF format
│   │   ├── Dataset.csv           # Converted CSV format
│   │   └── convert_arff.py       # ARFF to CSV converter
│   └── processed/                # Pre-split train/test data
│       ├── X_train.csv           # Training features (9,853 samples)
│       ├── X_test.csv            # Test features (2,464 samples)
│       ├── y_train.csv           # Training labels
│       └── y_test.csv            # Test labels
├── src/                          # Source code modules
│   ├── data_preprocessing.py     # Data loading, SMOTE balancing, splitting
│   ├── prepare_data.py           # One-time script to create train/test splits
│   ├── model_training.py         # Training utilities and hyperparameter search
│   ├── evaluate_models.py        # Evaluation metrics, confusion matrix, ROC curves
│   ├── feature_importance.py     # feature importance analysis
│   ├── run_lr.py                 # Train Logistic Regression
│   ├── run_svm.py                # Train SVM with hyperparameter tuning
│   ├── run_rf.py                 # Train Random Forest with hyperparameter tuning
│   ├── run_dnn.py                # Train Deep Neural Network (PyTorch)
│   └── run_ensemble.py           # Train Voting Ensemble (RF + SVM + DNN)
├── results/                      # Model outputs and visualizations
│   ├── metrics_summary.csv       # Performance metrics for all models
│   ├── neural_network.pth        # Trained PyTorch DNN weights
│   ├── confusion_matrices/       # Confusion matrix plots
│   ├── feature_importance/       # feature importance plots
│   └── roc_curves/               # ROC-AUC curves
├── report/                       # LaTeX report and figures
├── presentation/                 # Presentation slides
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/Nishaant-Soni/Phising_Website_Detection.git
cd Phising_Website_Detection
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n phishing python=3.9
conda activate phishing
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- pandas, numpy — Data manipulation
- scikit-learn — ML models and metrics
- imbalanced-learn — SMOTE for class balancing
- matplotlib, seaborn — Visualizations
- torch — Deep Neural Network

---

## Quick Start for replicating results (Full Pipeline)

To replicate all results in one go:

```bash
# 1. Train all models (generates confusion matrices & ROC curves automatically)
python src/run_lr.py         
python src/run_svm.py        
python src/run_rf.py         
python src/run_dnn.py         
python src/run_ensemble.py  

# 2. Generate feature importance plots
python src/feature_importance.py  # → results/feature_importance/*.png

# 3. View summary metrics
cat results/metrics_summary.csv
```

**Generated Visualizations:**
- **Confusion Matrices:** `results/confusion_matrices/` (created during each model training)
- **ROC Curves:** `results/roc_curves/` (created during each model training)
- **Feature Importance:** `results/feature_importance/` (created by feature_importance.py)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---