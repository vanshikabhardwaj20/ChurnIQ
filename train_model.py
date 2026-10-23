
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

DATA_PATH = Path("data/Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/churn_model.joblib")
IMPORTANCE_PATH = Path("models/feature_importance.csv")

data = pd.read_csv(DATA_PATH)

print("=" * 55)
print("CHURNIQ - MODEL TRAINING AND EXPLAINABILITY")
print("=" * 55)

# --------------------------------------------------
# 2. CLEAN DATA
# --------------------------------------------------

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)

data = data.dropna(subset=["TotalCharges"]).copy()
data = data.drop(columns=["customerID"])
data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

print("\nCustomers available:", len(data))
print("Missing values:", data.isnull().sum().sum())

# --------------------------------------------------
# 3. FEATURES AND TARGET
# --------------------------------------------------

X = data.drop(columns=["Churn"])
y = data["Churn"]

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    column for column in X.columns
    if column not in numeric_features
]

# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining customers:", len(X_train))
print("Testing customers:", len(X_test))

# --------------------------------------------------
# 5. PREPROCESSING
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# --------------------------------------------------
# 6. MODEL PIPELINE
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

print("\nTraining model...")
model.fit(X_train, y_train)
print("Training completed!")

# --------------------------------------------------
# 8. EVALUATION
# --------------------------------------------------

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)

print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, predictions, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, probabilities):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Stayed", "Churned"],
    zero_division=0
))

# --------------------------------------------------
# 9. EXPLAIN MODEL FEATURES
# --------------------------------------------------

print("\nGenerating feature importance...")

preprocessor_fitted = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor_fitted.get_feature_names_out()
coefficients = classifier.coef_[0]

importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "AbsoluteImportance": abs(coefficients)
})

# Positive coefficients increase predicted churn odds;
# negative coefficients decrease predicted churn odds.
importance["Direction"] = importance["Coefficient"].apply(
    lambda value: (
        "Associated with higher churn odds"
        if value > 0
        else "Associated with lower churn odds"
        if value < 0
        else "No linear effect"
    )
)

importance = importance.sort_values(
    "AbsoluteImportance",
    ascending=False
)

IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
importance.to_csv(IMPORTANCE_PATH, index=False)

print("\nTop 15 features by absolute coefficient:")
print(
    importance[
        ["Feature", "Coefficient", "Direction"]
    ].head(15).to_string(index=False)
)

# --------------------------------------------------
# 10. SAVE MODEL
# --------------------------------------------------

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
print("Feature importance saved to:", IMPORTANCE_PATH)
print("\nCHURNIQ TRAINING AND EXPLAINABILITY COMPLETE!")
