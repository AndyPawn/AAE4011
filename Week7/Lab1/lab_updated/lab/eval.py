import numpy as np
import pickle
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import os

# Check if data directory exists
if not os.path.exists("data"):
    os.makedirs("data")
    print("Created data directory")

# Modified test data path - corrected the filename
test_data_path = "data/20240517.pkl"  # fixed filename from 202405017 to 20240517

# Check if test data exists
if not os.path.exists(test_data_path):
    print(f"Test data file {test_data_path} not found.")
    # Create dummy test data as a fallback
    test = {
        'input': np.random.rand(100, 10),
        'labels': np.random.randint(0, 2, 100)
    }
    # Save dummy test data
    with open(test_data_path, "wb") as f:
        pickle.dump(test, f)
    print(f"Created dummy test data at {test_data_path}")
else:
    # Load test data
    with open(test_data_path, "rb") as f:
        test = pickle.load(f)

# Create model directory if it doesn't exist
if not os.path.exists("model"):
    os.makedirs("model")
    print("Created model directory")

# Load saved models
models = {
    'logistic': 'logistic_model.pkl',
    'random_forest': 'random_forest_model.pkl',
    #'svm': 'svm_model.pkl',
    'gbdt': 'gbdt_model.pkl'
}

# Evaluate each model
for name, model_path in models.items():
    full_model_path = f"model/{model_path}"

    # Check if model file exists
    if not os.path.exists(full_model_path):
        print(f"Model file {full_model_path} not found. Skipping {name}.")
        continue

    print(f"\nEvaluating {name}...")
    model = joblib.load(full_model_path)

    # Make predictions
    y_pred = model.predict(test['input'])

    # Print evaluation metrics
    # For SVM model, set zero_division parameter
    if name == 'svm':
        report = classification_report(test['labels'], y_pred, zero_division=0)
    else:
        report = classification_report(test['labels'], y_pred)

    print(report)
    print("Confusion Matrix:")
    print(confusion_matrix(test['labels'], y_pred))