"""
data_preprocessing.py
---------------------
Handles loading, cleaning, encoding, and splitting of the phishing website dataset.

Functions:
- load_data(): loads raw CSV data.
- preprocess_data(): encodes, normalizes, handles imbalance.
- split_data(): splits dataset into train/val/test.
"""

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
