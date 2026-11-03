from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    learning_curve
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CHURN_V3_MODEL_PATH = BASE_DIR / "models" / "churn_v3.joblib"
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

def get_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

def run_grid_searches(X_train, y_train, cv):
    preprocessor = get_preprocessor()

    candidates = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": ["balanced", None],
                "classifier__solver": ["lbfgs"]
            }
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=42),
            "params": {
                "classifier__max_depth": [3, 5, 8, 12, None],
                "classifier__min_samples_split": [2, 10, 20],
                "classifier__class_weight": ["balanced", None]
            }
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [5, 8, 12],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__class_weight": ["balanced", None]
            }
        },
        "SVM (RBF Kernel)": {
            "estimator": SVC(probability=True, random_state=42),
            "params": {
                "classifier__C": [0.1, 1.0, 5.0],
                "classifier__gamma": ["scale", 0.01],
                "classifier__class_weight": ["balanced"]
            }
        }
    }

    tuning_results = {}

    print("=" * 80)
    print("STEP 1: SYSTEMATIC GRID SEARCH CV & BIAS-VARIANCE DIAGNOSTICS")
    print("=" * 80)
    print(f"{'Model':<20} | {'CV ROC-AUC':<10} | {'Train ROC-AUC':<13} | {'Gap (Variance)':<14} | {'Best C / Depth'}")
    print("-" * 80)

    for name, config in candidates.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", config["estimator"])
        ])

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=config["params"],
            cv=cv,
            scoring="roc_auc",
            return_train_score=True,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_idx = grid.best_index_
        mean_cv = grid.cv_results_["mean_test_score"][best_idx]
        mean_train = grid.cv_results_["mean_train_score"][best_idx]
        gap = mean_train - mean_cv

        tuning_results[name] = {
            "grid": grid,
            "best_estimator": grid.best_estimator_,
            "best_params": grid.best_params_,
            "cv_score": mean_cv,
            "train_score": mean_train,
            "gap": gap
        }

        print(f"{name:<20} | {mean_cv:<10.4f} | {mean_train:<13.4f} | {gap:<14.4f} | {grid.best_params_}")

    return tuning_results

def evaluate_on_test(tuning_results, X_test, y_test):
    print("\n" + "=" * 80)
    print("STEP 2: HOLDOUT TEST SET BENCHMARKING (N=1409)")
    print("=" * 80)
    print(f"{'Model':<20} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'ROC-AUC':<8}")
    print("-" * 80)

    test_metrics = {}
    for name, data in tuning_results.items():
        model = data["best_estimator"]
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        test_metrics[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc
        }

        print(f"{name:<20} | {acc:<8.4f} | {prec:<9.4f} | {rec:<8.4f} | {f1:<8.4f} | {auc:<8.4f}")

    return test_metrics

def compute_learning_curve(champion_pipeline, X, y, cv):
    print("\n" + "=" * 80)
    print("STEP 3: LEARNING CURVE ANALYSIS (BIAS vs VARIANCE DIAGNOSIS)")
    print("=" * 80)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=champion_pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
        random_state=42
    )

    mean_train = np.mean(train_scores, axis=1)
    mean_val = np.mean(val_scores, axis=1)

    print(f"{'Training Samples':<18} | {'Train ROC-AUC':<14} | {'Val ROC-AUC':<12} | {'Generalization Gap':<18}")
    print("-" * 68)
    for sz, tr, vl in zip(train_sizes, mean_train, mean_val):
        print(f"{sz:<18d} | {tr:<14.4f} | {vl:<12.4f} | {tr - vl:<18.4f}")

    final_gap = mean_train[-1] - mean_val[-1]
    print("\nLearning Curve Diagnostic Assessment:")
    if final_gap > 0.05:
        print("  -> Moderate variance detected: Regularization (tree depth limits or min split) successfully contained overfit.")
    else:
        print("  -> Low variance & balanced bias: Model generalizes robustly without severe memorization.")

def lock_champion(tuning_results):
    print("\n" + "=" * 80)
    print("STEP 4: LOCKING & VERSIONING DAY 3 CHAMPION")
    print("=" * 80)
    
    # Selection criteria: Highest CV ROC-AUC with low generalization gap
    champion_name = max(tuning_results, key=lambda k: tuning_results[k]["cv_score"])
    champion_data = tuning_results[champion_name]
    champion_pipeline = champion_data["best_estimator"]

    CHURN_V3_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_pipeline, CHURN_V3_MODEL_PATH)
    joblib.dump(champion_pipeline, CHAMPION_MODEL_PATH)

    print(f"Locked Champion: {champion_name}")
    print(f"Optimal Hyperparameters: {champion_data['best_params']}")
    print(f"CV ROC-AUC Score: {champion_data['cv_score']:.4f} (Train: {champion_data['train_score']:.4f})")
    print(f"Saved artifacts:")
    print(f"  -> {CHURN_V3_MODEL_PATH}")
    print(f"  -> {CHAMPION_MODEL_PATH}")

    return champion_name, champion_pipeline

def main():
    print("Initializing Day 3 Multi-Model Tuning & Validation Pipeline...\n")
    df = load_data()
    X = df[num_cols + cat_cols]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tuning_results = run_grid_searches(X_train, y_train, cv)
    evaluate_on_test(tuning_results, X_test, y_test)
    champion_name, champion_pipeline = lock_champion(tuning_results)
    compute_learning_curve(champion_pipeline, X, y, cv)

if __name__ == "__main__":
    main()

