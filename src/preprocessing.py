# Centrailised Dataset Preprocessing

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
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

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0}).astype(int)
    return df

def get_train_test_data(path: Path = DATA_PATH, test_size: float = 0.2, random_state: int = 42, strategy: str = "median"):
    df = load_data(path)
    X = df[NUMERICAL_COLS + CATEGORICAL_COLS]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    fill_val = X_train["TotalCharges"].median() if strategy == "median" else X_train["TotalCharges"].mean()
    X_train["TotalCharges"] = X_train["TotalCharges"].fillna(fill_val)
    X_test["TotalCharges"] = X_test["TotalCharges"].fillna(fill_val)

    return X_train, X_test, y_train, y_test

def get_preprocessor(
    num_columns=None,
    cat_columns=None,
    sparse_output: bool = False,
    handle_unknown: str = "ignore",
    impute_strategy: str = "median",
) -> ColumnTransformer:
    if num_columns is None:
        num_columns = NUMERICAL_COLS
    if cat_columns is None:
        cat_columns = CATEGORICAL_COLS

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        ("scaler", StandardScaler()),
    ])

    encoder = OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse_output)

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_columns),
            ("cat", encoder, cat_columns),
        ]
    )
