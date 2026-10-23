# ChurnIQ — Customer Retention Intelligence Platform

ChurnIQ is a machine-learning-based customer retention platform that predicts the likelihood of a telecom customer leaving a company. It helps businesses identify customers who may be at risk and understand factors associated with customer churn.

## Features

- **Customer Churn Prediction:** Estimates the probability that a customer will leave.
- **Risk Classification:** Categorizes customers into Low, Medium, or High risk.
- **Personalized Recommendations:** Provides suggestions based on customer characteristics, such as contract type, tenure, and subscribed services.
- **Analytics Dashboard:** Displays customer statistics, churn patterns, and important churn-related factors.
- **Model Evaluation:** Reports accuracy, precision, recall, F1-score, and ROC-AUC.
- **Input Validation:** Checks submitted customer information before generating predictions.
- **Automated Testing:** Uses pytest to test key application routes and prediction functionality.

## Technology Stack

- Python
- Flask
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- HTML, CSS, and JavaScript
- Chart.js
- Pytest

## Machine Learning Approach

ChurnIQ uses a Scikit-learn machine-learning pipeline with preprocessing and Logistic Regression.

The model estimates churn probability from customer information, including tenure, contract type, monthly charges, and subscribed services.

The project also compares Logistic Regression with Random Forest to examine differences in their evaluation metrics.

### Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | 72.57% |
| Precision | 49.01% |
| Recall | 79.68% |
| F1-score | 60.69% |
| ROC-AUC | 83.51% |

These results are from the project's current evaluation run. Recall measures how many actual churners the model identifies, while precision measures how many customers flagged as churners actually churned in the evaluation data.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

- Original dataset: 7,043 customer records
- Records remaining after cleaning: 7,032
- Records with missing `TotalCharges` values removed: 11
- Target variable: `Churn`

The dataset is used for experimentation and evaluation. Performance on other companies' customers may differ and would require appropriate data preparation, validation, and potentially model retraining.

## Project Structure

```text
CHURN IQ/
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   ├── churn_model.joblib
│   ├── feature_importance.csv
│   └── model_metrics.json
├── notebooks/
├── static/
├── templates/
│   ├── index.html
│   └── dashboard.html
├── tests/
│   └── test_app.py
├── app.py
├── train_model.py
├── evaluate_model.py
├── compare_models.py
├── dashboard.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "CHURN IQ"
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

The analytics dashboard is available at:

http://127.0.0.1:5000/dashboard

## Run Tests

Run the automated test suite using:

```bash
python -m pytest -v
```

The current test suite has five passing tests.

## Important Limitations

- Predictions indicate estimated risk, not certainty that a customer will leave.
- Recommendations are suggestions; their effectiveness has not yet been established through a real-world intervention study.
- The model was evaluated on a particular telecom dataset and may not generalize to other customer populations without further validation.
- The project is a decision-support prototype, not a guarantee of customer retention.

## Future Improvements

- Add model calibration and threshold tuning.
- Improve explainability with customer-level feature explanations.
- Add historical churn trend analysis.
- Evaluate recommendations through controlled experiments.
- Support uploading new customer datasets with validation.
- Improve deployment, monitoring, and model-drift detection.

## Author

Developed as a machine-learning and data analytics project focused on customer churn prediction and retention insights.
