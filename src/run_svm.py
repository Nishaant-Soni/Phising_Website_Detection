from data_preprocessing import load_data, preprocess_data, split_data
from model_training import build_models, train_model
from evaluate_models import evaluate, save_results

# 1. Data
df = load_data("./data/raw/Training_Dataset.csv")
X_res, y_res = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X_res, y_res, test_size=0.2)

# 2. Build models
all_models = build_models()
metrics_list = []

model = all_models["SVM (RBF)"]
# 3. Train (with tuning) & save
best_model = train_model(model, X_train, y_train, "SVM (RBF)")

# 4. Evaluate
metrics = evaluate(best_model, X_test, y_test, "SVM (RBF)")
metrics_list.append(metrics)

# 5. Save metrics summary (optional)
save_results(metrics_list)
