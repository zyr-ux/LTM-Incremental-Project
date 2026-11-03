# Prediction Engine

from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_CANDIDATES = [
    BASE_DIR / "models" / "churn_v3.joblib",
    BASE_DIR / "models" / "churn_pipeline_champion.joblib",
    BASE_DIR / "models" / "churn_pipeline_baseline.joblib",
]

_model = None

def get_model_path():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No model found in {[str(p) for p in MODEL_CANDIDATES]}. Run pipeline.py first."
    )

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(get_model_path())
    return _model

def _get_risk_level(prob):
    if prob >= 0.7:
        return "High"
    if prob >= 0.4:
        return "Medium"
    return "Low"

def predict_churn(customer_data):
    clf = get_model()

    single_input = isinstance(customer_data, dict)
    if single_input:
        df = pd.DataFrame([customer_data])
    elif isinstance(customer_data, list):
        df = pd.DataFrame(customer_data)
    else:
        df = customer_data.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype(str).str.strip(), errors="coerce"
        ).fillna(0.0)

    preds = clf.predict(df)
    probs = clf.predict_proba(df)[:, 1]

    results = [
        {
            "churn_prediction": int(pred),
            "churn_probability": round(float(prob), 4),
            "risk_level": _get_risk_level(float(prob)),
        }
        for pred, prob in zip(preds, probs)
    ]

    return results[0] if single_input else results

if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }
    print("Testing prediction service:")
    print(predict_churn(sample_customer))
