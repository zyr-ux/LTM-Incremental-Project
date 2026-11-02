# Prediction Engine

from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
CHAMPION_MODEL_PATH = BASE_DIR / "models" / "churn_pipeline_champion.joblib"
BASELINE_MODEL_PATH = BASE_DIR / "models" / "churn_pipeline_baseline.joblib"
MODEL_PATH = CHAMPION_MODEL_PATH if CHAMPION_MODEL_PATH.exists() else BASELINE_MODEL_PATH

# Cache loaded model in memory
model = None

def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run pipeline.py first.")
        model = joblib.load(MODEL_PATH)
    return model

def predict_churn(customer_data):
    clf = get_model()

    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
        single_input = True
    elif isinstance(customer_data, list):
        df = pd.DataFrame(customer_data)
        single_input = False
    else:
        df = customer_data.copy()
        single_input = False

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)

    preds = clf.predict(df)
    probs = clf.predict_proba(df)[:, 1]

    results = []
    for pred, prob in zip(preds, probs):
        prob_val = round(float(prob), 4)
        if prob_val >= 0.7:
            risk = "High"
        elif prob_val >= 0.4:
            risk = "Medium"
        else:
            risk = "Low"

        results.append({
            "churn_prediction": int(pred),
            "churn_probability": prob_val,
            "risk_level": risk
        })

    if single_input:
        return results[0]
    return results

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
