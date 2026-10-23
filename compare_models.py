
import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# 1. Create output folder
os.makedirs("models", exist_ok=True)


# 2. Load and clean the dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.dropna(subset=["TotalCharges"])
df = df.drop(columns=["customerID"])

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

X = df.drop(columns=["Churn"])
y = df["Churn"]


# 3. Identify numerical and categorical columns
numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    col for col in X.columns
    if col not in numerical_features
]


# 4. Define preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# 5. Create one shared train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 6. Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# 7. Train and evaluate
results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", clone(model))
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Confusion matrix:
    # [[True Negative, False Positive],
    #  [False Negative, True Positive]]
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    scores = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Specificity": specificity,
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp
    }

    results.append(scores)

    print(f"\n{name}")
    print("-" * 45)

    for metric, value in scores.items():
        if metric != "Model":
            if isinstance(value, float):
                print(f"{metric:20}: {value:.4f}")
            else:
                print(f"{metric:20}: {value}")

    print("\nConfusion Matrix:")
    print([[tn, fp], [fn, tp]])


# 8. Display and save the comparison
comparison = pd.DataFrame(results).set_index("Model")

print("\n\nFINAL MODEL COMPARISON")
print("=" * 80)
print(comparison.round(4).to_string())

comparison.to_csv("models/model_comparison.csv")
print("\nSaved: models/model_comparison.csv")


# 9. Plot key metrics for both models
metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1 Score",
    "ROC-AUC"
]

plot_data = comparison[metric_columns]

ax = plot_data.T.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1),
    width=0.75
)

ax.set_title("ChurnIQ: Model Performance Comparison")
ax.set_xlabel("Evaluation Metric")
ax.set_ylabel("Score")
ax.legend(title="Model")
ax.grid(axis="y", linestyle="--", alpha=0.35)
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "models/model_comparison.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("Saved: models/model_comparison.png")
print("\nComparison complete.")
print("Your saved production model has not been changed.")
