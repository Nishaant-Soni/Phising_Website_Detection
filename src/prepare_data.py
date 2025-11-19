"""
prepare_data.py
---------------
One-time script to create and save train/test splits.
Run this once before training models to ensure all models use the same data.

Usage:
    python src/prepare_data.py
"""

from data_preprocessing import load_data, preprocess_data, split_data, save_split_data

def main():
    print("Loading and preprocessing data...")
    df = load_data("./data/raw/Dataset.csv")
    X_res, y_res = preprocess_data(df)
    
    print("\nCreating 80-20 train/test split...")
    X_train, X_test, y_train, y_test = split_data(X_res, y_res, test_size=0.2)
    
    print("\nSaving splits to data/processed/...")
    save_split_data(X_train, X_test, y_train, y_test)
    
    print("\n✓ Data preparation complete!")
    print("You can now run model training scripts (run_lr.py, run_svm.py, etc.)")

if __name__ == "__main__":
    main()
