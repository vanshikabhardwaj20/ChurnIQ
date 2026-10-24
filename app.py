
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import json
import math

app = Flask(__name__)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "models", "feature_importance.csv")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")
COMPARISON_PATH = os.path.join(BASE_DIR, "models", "model_comparison.csv")

# Load the trained machine-learning model
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html")

    try:


        # 1. Read customer information from the form
        customer = {
            "gender": request.form["gender"].strip(),
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"].strip(),
            "Dependents": request.form["Dependents"].strip(),
            "tenure": int(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"].strip(),
            "MultipleLines": request.form["MultipleLines"].strip(),
            "InternetService": request.form["InternetService"].strip(),
            "OnlineSecurity": request.form["OnlineSecurity"].strip(),
            "OnlineBackup": request.form["OnlineBackup"].strip(),
            "DeviceProtection": request.form["DeviceProtection"].strip(),
            "TechSupport": request.form["TechSupport"].strip(),
            "StreamingTV": request.form["StreamingTV"].strip(),
            "StreamingMovies": request.form["StreamingMovies"].strip(),
            "Contract": request.form["Contract"].strip(),
            "PaperlessBilling": request.form["PaperlessBilling"].strip(),
            "PaymentMethod": request.form["PaymentMethod"].strip(),
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"])
        }

        # 2. Validate categorical values
        allowed_values = {
            "gender": {"Female", "Male"},
            "Partner": {"Yes", "No"},
            "Dependents": {"Yes", "No"},
            "PhoneService": {"Yes", "No"},
            "MultipleLines": {"Yes", "No", "No phone service"},
            "InternetService": {"DSL", "Fiber optic", "No"},
            "OnlineSecurity": {"Yes", "No", "No internet service"},
            "OnlineBackup": {"Yes", "No", "No internet service"},
            "DeviceProtection": {"Yes", "No", "No internet service"},
            "TechSupport": {"Yes", "No", "No internet service"},
            "StreamingTV": {"Yes", "No", "No internet service"},
            "StreamingMovies": {"Yes", "No", "No internet service"},
            "Contract": {"Month-to-month", "One year", "Two year"},
            "PaperlessBilling": {"Yes", "No"},
            "PaymentMethod": {
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            }
        }

        for field, valid_options in allowed_values.items():
            if customer[field] not in valid_options:
                raise ValueError(f"Invalid value for {field}")

        # 3. Validate numerical values
        if customer["SeniorCitizen"] not in (0, 1):
            raise ValueError("SeniorCitizen must be 0 or 1")

        if not 0 <= customer["tenure"] <= 72:
            raise ValueError("Tenure must be between 0 and 72 months")

        for field in ("MonthlyCharges", "TotalCharges"):
            value = customer[field]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid value for {field}")

        # 4. Prepare customer data
        customer_df = pd.DataFrame([customer])

        # 5. Predict churn probability
        churn_probability = model.predict_proba(customer_df)[0][1]
        churn_percent = round(churn_probability * 100, 2)

        # 6. Assign risk category
        if churn_probability >= 0.70:
            risk = "High"
        elif churn_probability >= 0.40:
            risk = "Medium"
        else:
            risk = "Low"

        # 7. Generate retention recommendations
        recommendations = []

        if customer["Contract"] == "Month-to-month":
            recommendations.append(
                "Offer a suitable longer-term contract option."
            )

        if customer["tenure"] <= 12:
            recommendations.append(
                "Provide extra onboarding support to this newer customer."
            )

        if customer["TechSupport"] == "No":
            recommendations.append(
                "Consider offering technical support options."
            )

        if customer["OnlineSecurity"] == "No":
            recommendations.append(
                "Explain the available online security options."
            )

        if not recommendations:
            recommendations.append(
                "Monitor customer satisfaction and review service usage."
            )

        prediction = (
            "Likely to Churn"
            if churn_probability >= 0.50
            else "Likely to Stay"
        )

        return render_template(
            "index.html",
            prediction=prediction,
            churn_percent=churn_percent,
            risk=risk,
            recommendations=recommendations,
            customer=customer
        )

    except (ValueError, KeyError, OverflowError):
        return render_template(
            "index.html",
            error=(
                "Invalid customer details. "
                "Please check all fields and try again."
            )
        ), 400


@app.route("/dashboard")
def analytics_dashboard():
    # 1. Load and clean dataset
    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    df = df.dropna(subset=["TotalCharges"])

    # 2. Calculate customer summary metrics
    total_customers = len(df)
    churned = int((df["Churn"] == "Yes").sum())
    retained = total_customers - churned

    churn_rate = (
        round(churned / total_customers * 100, 2)
        if total_customers > 0
        else 0
    )

    # 3. Churn rate by contract type
    contract_churn = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: round((x == "Yes").mean() * 100, 2))
    )

    # 4. Churn rate by tenure group
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    tenure_churn = (
        df.groupby("TenureGroup", observed=False)["Churn"]
        .apply(lambda x: round((x == "Yes").mean() * 100, 2))
    )

    # 5. Load and prepare feature coefficients
    importance_df = pd.read_csv(IMPORTANCE_PATH)

    importance_df["AbsoluteCoefficient"] = (
        importance_df["Coefficient"].abs()
    )

    top_features = (
        importance_df.sort_values(
            "AbsoluteCoefficient", ascending=False
        )
        .head(10)
        .copy()
    )

    top_features["Feature"] = (
        top_features["Feature"]
        .str.replace("categorical__", "", regex=False)
        .str.replace("numeric__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    feature_labels = top_features["Feature"].tolist()
    feature_values = top_features["Coefficient"].round(3).tolist()

    # 6. Load current production-model evaluation metrics
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        model_metrics = json.load(file)

    # 7. Load model comparison data
    comparison_df = pd.read_csv(COMPARISON_PATH)

    required_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1 Score",
        "ROC-AUC"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in comparison_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in model_comparison.csv: {missing_columns}"
        )

    comparison_labels = comparison_df["Model"].tolist()

    comparison_metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1 Score",
        "ROC-AUC"
    ]

    comparison_data = {
        metric: [
            round(float(value), 4)
            for value in comparison_df[metric].tolist()
        ]
        for metric in comparison_metrics
    }

    # 8. Pass data to dashboard.html
    return render_template(
        "dashboard.html",
        total_customers=total_customers,
        churned=churned,
        retained=retained,
        churn_rate=churn_rate,

        # Contract chart
        contract_labels=contract_churn.index.tolist(),
        contract_values=contract_churn.tolist(),
        contract_rates=contract_churn.tolist(),

        # Tenure chart
        tenure_labels=[str(x) for x in tenure_churn.index],
        tenure_values=tenure_churn.tolist(),
        tenure_rates=tenure_churn.tolist(),

        # Churn factors chart
        feature_labels=feature_labels,
        feature_values=feature_values,
        factor_labels=feature_labels,
        factor_values=feature_values,

        # Production-model metrics
        model_metrics=model_metrics,

        # Model comparison chart data
        comparison_labels=comparison_labels,
        comparison_metrics=comparison_metrics,
        comparison_data=comparison_data
    )


if __name__ == "__main__":
    app.run(debug=True)
