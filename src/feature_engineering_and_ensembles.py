from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CHURN_V4_MODEL_PATH = BASE_DIR / "models" / "churn_v4.joblib"
CHAMPION_MODEL_PATH = BASE_DIR / "models" / "churn_pipeline_champion.joblib"

RAW_NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
RAW_CAT_COLS = [
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
    "PaymentMethod",
]

NUM_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "monthly_to_total_ratio",
    "charge_ratio",
    "avg_monthly_diff",
    "total_services",
]

FLAG_COLS = [
    "has_streaming",
    "has_security_or_backup",
    "is_month_to_month",
    "has_fiber_optic",
    "electronic_check",
]

CAT_COLS = RAW_CAT_COLS + ["tenure_group"]


class RetailFeatureEngineer(BaseEstimator, TransformerMixin):
    __module__ = "src.feature_engineering_and_ensembles"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(
                X["TotalCharges"].astype(str).str.strip(), errors="coerce"
            ).fillna(0.0)

        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[-1, 12, 24, 48, 60, 100],
            labels=["0-12", "12-24", "24-48", "48-60", "60+"],
        ).astype(str)

        X["monthly_to_total_ratio"] = X["MonthlyCharges"] / (X["TotalCharges"] + 1.0)
        X["charge_ratio"] = X["TotalCharges"] / (X["MonthlyCharges"] * X["tenure"] + 1e-5)
        X["avg_monthly_diff"] = X["MonthlyCharges"] - (
            X["TotalCharges"] / (X["tenure"] + 1.0)
        )

        service_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        X["total_services"] = (X[service_cols] == "Yes").sum(axis=1)
        X["has_streaming"] = (
            (X["StreamingTV"] == "Yes") | (X["StreamingMovies"] == "Yes")
        ).astype(int)
        X["has_security_or_backup"] = (
            (X["OnlineSecurity"] == "Yes") | (X["OnlineBackup"] == "Yes")
        ).astype(int)

        X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
        X["has_fiber_optic"] = (X["InternetService"] == "Fiber optic").astype(int)
        X["electronic_check"] = (
            X["PaymentMethod"] == "Electronic check"
        ).astype(int)

        return X


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    ).fillna(0.0)
    df["Churn"] = (
        df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0}).astype(int)
    )
    return df


def get_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("passthrough", "passthrough", FLAG_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )


def evaluate_models(X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preprocessor = get_preprocessor()

    candidates = {
        "Day-3 Champion (RF baseline)": Pipeline([
            ("prep", ColumnTransformer([
                ("num", StandardScaler(), RAW_NUM_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), RAW_CAT_COLS),
            ])),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=10, random_state=42)),
        ]),
        "Random Forest (Day 4 FE)": Pipeline([
            ("fe", RetailFeatureEngineer()),
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=10, random_state=42)),
        ]),
        "Random Forest + PCA (15 comps)": Pipeline([
            ("fe", RetailFeatureEngineer()),
            ("prep", preprocessor),
            ("pca", PCA(n_components=15, random_state=42)),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=10, random_state=42)),
        ]),
        "XGBoost (Day 4 FE Champion)": Pipeline([
            ("fe", RetailFeatureEngineer()),
            ("prep", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )),
        ]),
        "XGBoost + PCA (15 comps)": Pipeline([
            ("fe", RetailFeatureEngineer()),
            ("prep", preprocessor),
            ("pca", PCA(n_components=15, random_state=42)),
            ("clf", XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )),
        ]),
    }

    results = []
    print("\n" + "=" * 80)
    print("DAY 4: FEATURE ENGINEERING & ENSEMBLE EVALUATION")
    print("=" * 80)

    for name, pipe in candidates.items():
        print(f"\nEvaluating: {name} ...")
        fit_X_train = X_train
        fit_X_test = X_test

        cv_auc = cross_val_score(pipe, fit_X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(fit_X_train, y_train)

        train_prob = pipe.predict_proba(fit_X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_prob)

        preds = pipe.predict(fit_X_test)
        probs = pipe.predict_proba(fit_X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        gap = train_auc - cv_auc.mean()

        results.append({
            "model_name": name,
            "pipeline": pipe,
            "cv_roc_auc_mean": cv_auc.mean(),
            "cv_roc_auc_std": cv_auc.std(),
            "train_roc_auc": train_auc,
            "variance_gap": gap,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
        })

        print(f"  CV ROC-AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f}) | Train AUC: {train_auc:.4f} | Gap: {gap:.4f}")
        print(f"  Holdout Test -> Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}% | AUC: {auc*100:.2f}%")

    return results


def main():
    df = load_data()
    raw_feature_cols = RAW_NUM_COLS + RAW_CAT_COLS
    X = df[raw_feature_cols]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = evaluate_models(X_train, y_train, X_test, y_test)

    champion_res = next(r for r in results if r["model_name"] == "XGBoost (Day 4 FE Champion)")
    champion_pipeline = champion_res["pipeline"]

    print("\n" + "=" * 80)
    print("FINALIZING DAY 4 PRODUCTION CHAMPION")
    print("=" * 80)
    print(f"Champion: {champion_res['model_name']}")
    print(f"CV ROC-AUC: {champion_res['cv_roc_auc_mean']:.4f}")
    print(f"Test ROC-AUC: {champion_res['roc_auc']:.4f}")
    print(f"Test Precision: {champion_res['precision']*100:.2f}%")
    print(f"Test Recall: {champion_res['recall']*100:.2f}%")

    preds = champion_pipeline.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, preds))

    CHURN_V4_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_pipeline, CHURN_V4_MODEL_PATH)
    joblib.dump(champion_pipeline, CHAMPION_MODEL_PATH)

    print(f"Successfully saved Day 4 champion model to: {CHURN_V4_MODEL_PATH}")
    print(f"Successfully refreshed active champion model at: {CHAMPION_MODEL_PATH}")


if __name__ == "__main__":
    main()

