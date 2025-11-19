"""
data_preprocessing.py
---------------------
Handles loading, cleaning, encoding, and splitting of the phishing website dataset.

Functions:
- load_data(): loads raw CSV data.
- preprocess_data(): encodes, normalizes, handles imbalance.
- split_data(): splits dataset into train/val/test.
- save_split_data(): saves train/test splits to processed folder.
- load_split_data(): loads pre-split train/test data.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(path: str) -> pd.DataFrame:
    """Load the phishing dataset from CSV."""
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame):
    """Apply label encoding, SMOTE balancing, and return features/labels."""
    X = df.drop('Result', axis=1)
    y = df['Result']
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def split_data(X, y, test_size=0.2):
    """Split dataset into train and test sets."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

def save_split_data(X_train, X_test, y_train, y_test, output_dir="./data/processed"):
    """Save train/test splits to CSV files in the processed folder."""
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False, header=True)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header=True)
    
    print(f"Train/test data saved to {output_dir}/")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - y_test: {y_test.shape}")

def load_split_data(input_dir="./data/processed"):
    """Load pre-split train/test data from CSV files."""
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv")).squeeze()
    
    print(f"Loaded train/test data from {input_dir}/")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test
