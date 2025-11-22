"""
model_training.py
-----------------
Implements ML models: Logistic Regression, SVM, Random Forest, PyTorch DNN, Ensemble.
Performs hyperparameter tuning and saves trained models.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler


class PyTorchDNN(nn.Module):
    """PyTorch Deep Neural Network for binary classification."""
    
    def __init__(self, input_size, hidden_layers=[64, 32], dropout_rate=0.2):
        super(PyTorchDNN, self).__init__()
        self.scaler = StandardScaler()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
    def fit(self, X, y, epochs=100, batch_size=32, lr=0.001, verbose=False):
        """Train the model."""
        X_scaled = self.scaler.fit_transform(X)
        y_binary = np.where(y.values == -1, 0, 1)
        
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_binary.reshape(-1, 1))
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    def predict(self, X):
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            outputs = self(X_tensor)
            # Convert 0/1 predictions back to -1/1 format to match original labels
            predictions = (outputs.numpy() > 0.5).astype(int).flatten()
            predictions = np.where(predictions == 0, -1, 1)
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        self.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            outputs = self(X_tensor).numpy().flatten()
        return np.column_stack([1 - outputs, outputs])


def build_models():
    """Initialize multiple models for comparison."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",     
            random_state=42,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            probability=True,        
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "Neural Network": None,  # Will be created dynamically based on input size
    }

    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting="soft"
    )
    models["Ensemble"] = ensemble
    return models


def train_model(model, X_train, y_train, model_name):
    """
    Train (with hyperparameter tuning for LR and SVM) and save the model.
    Returns the best model.
    """
    best_model = model

    # --- Logistic Regression tuning ---
    if model_name == "Logistic Regression":
        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],   # if you want L1, switch solver to 'saga'
        }
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        print("Best params for Logistic Regression:", grid.best_params_)
        best_model = grid.best_estimator_

    # --- SVM (RBF) tuning ---
    elif model_name == "SVM (RBF)":
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", 0.01, 0.1, 1],
        }
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        print("Best params for SVM (RBF):", grid.best_params_)
        best_model = grid.best_estimator_
 
    # --- Random Forest tuning ---
    elif model_name == "Random Forest":
        param_grid = {
            "n_estimators": [80, 90, 100, 150, 180, 200, 250],
            "max_depth": [10, 20, 30, 35, 40]
    
        }

        grid = GridSearchCV(
            estimator = model,
            param_grid= param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(X_train, y_train)
        print("Best params for Random Forest:", grid.best_params_)
        best_model = grid.best_estimator_

    # -- PyTorch DNN training and tuning ---
    elif model_name == "Neural Network":
        input_size = X_train.shape[1]
        
        architectures = [
            [64, 32],
            [128, 64],
            [256, 128, 64],
            [128, 64, 32]
        ]
        
        best_f1 = 0
        best_config = None
        
        print("Training PyTorch DNN with different architectures...")
        for arch in architectures:
            dnn_model = PyTorchDNN(input_size, hidden_layers=arch)
            dnn_model.fit(X_train, y_train, epochs=150, verbose=False)
            
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import f1_score
            X_val_split, _, y_val_split, _ = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            y_val_pred = dnn_model.predict(X_val_split)
            f1 = f1_score(y_val_split, y_val_pred)
            
            print(f"Architecture {arch}: F1 = {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_config = arch
                best_model = dnn_model
        
        print(f"Best architecture: {best_config} with F1 = {best_f1:.4f}")
        
        best_model = PyTorchDNN(input_size, hidden_layers=best_config)
        best_model.fit(X_train, y_train, epochs=200, verbose=True)
    else:
        # Other models: just fit directly (RF, NN, Ensemble)
        best_model.fit(X_train, y_train)

    if model_name == "Neural Network":
        torch.save(best_model.state_dict(), f"results/{model_name.replace(' ', '_').lower()}.pth")
        joblib.dump(best_model, f"results/{model_name.replace(' ', '_').lower()}.pkl")
    else:
        joblib.dump(best_model, f"results/{model_name.replace(' ', '_').lower()}.pkl")
    return best_model

