
import json
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# File paths
DATA_PATH = "data/Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_model.joblib"
OUTPUT_PATH = "models/model_metrics.json"

# Load and clean the dataset
df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna(subset=["TotalCharges"])

# Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Re-create the test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load the already-trained model
model = joblib.load(MODEL_PATH)

# Generate predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
metrics = {
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "precision": round(
        float(precision_score(y_test, y_pred, zero_division=0)), 4
    ),
    "recall": round(float(recall_score(y_test, y_pred)), 4),
    "f1_score": round(float(f1_score(y_test, y_pred)), 4),
    "roc_auc": round(float(roc_auc_score(y_test, y_probability)), 4),
    "confusion_matrix": confusion_matrix(
        y_test, y_pred
    ).tolist()
}

# Save metrics to a JSON file
os.makedirs("models", exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

# Display the results
print("\nCHURNIQ MODEL EVALUATION")
print("-" * 35)

for metric, value in metrics.items():
    print(f"{metric}: {value}")

print(f"\nMetrics saved to: {OUTPUT_PATH}")
