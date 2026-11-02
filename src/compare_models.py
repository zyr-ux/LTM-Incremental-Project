from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CHAMPION_MODEL_PATH = BASE_DIR / "models" / "churn_pipeline_champion.joblib"

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
    df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0}).astype(int)
    return df

def compare_models():
    df = load_data()
    X = df[num_cols + cat_cols]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    candidate_models = {
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "SVM (RBF)": SVC(class_weight="balanced", probability=True, random_state=42)
    }

    results = {}
    fitted_pipelines = {}

    print(f"{'Model':<15} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'ROC-AUC':<8}")
    print("-" * 68)

    for name, clf in candidate_models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc
        }
        fitted_pipelines[name] = pipe

        print(f"{name:<15} | {acc:<8.4f} | {prec:<9.4f} | {rec:<8.4f} | {f1:<8.4f} | {auc:<8.4f}")

    # Select champion based on highest recall for churn prevention
    champion_name = max(results, key=lambda k: results[k]["recall"])
    champion_pipe = fitted_pipelines[champion_name]

    CHAMPION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_pipe, CHAMPION_MODEL_PATH)
    print(f"\nChampion model selected: {champion_name} (Recall: {results[champion_name]['recall']:.4f})")
    print(f"Saved champion pipeline to: {CHAMPION_MODEL_PATH}")

    return results

if __name__ == "__main__":
    compare_models()

