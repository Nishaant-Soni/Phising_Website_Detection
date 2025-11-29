"""
run_dnn.py
-----------
Train and evaluate a PyTorch Deep Neural Network (DNN) for phishing website detection
with comprehensive hyperparameter tuning and architecture search.

This script:
1. Loads pre-split train/test data from data/processed/
2. Performs extensive search over multiple hyperparameters
3. Identifies the best model configuration based on validation F1-score
4. Retrains the best model on full training data with extended epochs
5. Evaluates the final model on the test set
6. Saves performance metrics to results/metrics_summary.csv (appends if exists)
7. Generates and saves a confusion matrix visualization

Hyperparameter Grid Search:
    - Network Architectures: [64,32], [128,64], [256,128,64], [128,64,32]
    - Dropout Rates: 0.1, 0.2, 0.3
    - Learning Rates: 1e-3, 2e-3, 5e-4
    - Batch Sizes: 32, 64, 128

Training Process:
    1. Grid Search Phase:
       - 80/20 train/validation split (stratified)
       - Each configuration trained for 100 epochs
       - F1-score evaluated on validation set
       - Best hyperparameters identified
    
    2. Final Training Phase:
       - Best model retrained on full training data (100%)
       - Extended training: 150 epochs
       - Uses optimal hyperparameters from grid search
       - Maximizes use of available training data

Features:
    - Automatic feature scaling (StandardScaler)
    - Dropout regularization (prevents overfitting)
    - Adam optimizer with configurable learning rate
    - Binary cross-entropy loss
    - Early stopping consideration through validation monitoring
    - Stratified sampling for balanced validation

Outputs:
    - PyTorch model state: results/neural_network.pth
    - Full model object: results/neural_network.pkl (includes scaler)
    - Metrics appended to results/metrics_summary.csv
    - Confusion matrix: results/confusion_matrices/neural_network.png

Model Architecture:
    - Input layer: 30 features
    - Hidden layers: Variable (determined by grid search)
    - Activation: ReLU
    - Dropout: After each hidden layer
    - Output layer: Single neuron with sigmoid activation
    - Loss function: Binary Cross-Entropy
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
