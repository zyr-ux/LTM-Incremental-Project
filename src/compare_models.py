# Comparing Various Models

from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocessing import (
    get_train_test_data,
    get_preprocessor,
    BASE_DIR,
)

CHAMPION_MODEL_PATH = BASE_DIR / "models" / "churn_pipeline_champion.joblib"

def compare_models():
    X_train, X_test, y_train, y_test = get_train_test_data()

    preprocessor = get_preprocessor(handle_unknown="ignore", impute_strategy="median")

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

    champion_name = max(results, key=lambda k: results[k]["recall"])
    champion_pipeline = fitted_pipelines[champion_name]

    print("\n" + "=" * 68)
    print(f"Champion Selected: {champion_name} (Recall: {results[champion_name]['recall']:.4f})")
    print("=" * 68)

    CHAMPION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_pipeline, CHAMPION_MODEL_PATH)
    print(f"Saved Champion Pipeline to: {CHAMPION_MODEL_PATH}")

    return champion_name, results

if __name__ == "__main__":
    compare_models()
