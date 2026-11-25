import pandas as pd
from src.preprocessing import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    get_preprocessor,
    get_train_test_data,
    load_data,
)
from src.service import _get_risk_level, get_model_path, predict_churn


def test_load_data_cleans_targets_and_charges():
    df = load_data()
    assert not df.empty
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert set(df["Churn"].unique()).issubset({0, 1})


def test_train_test_split_has_no_nulls_and_expected_columns():
    X_train, X_test, y_train, y_test = get_train_test_data()
    expected_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    assert list(X_train.columns) == expected_cols
    assert list(X_test.columns) == expected_cols
    assert int(X_train["TotalCharges"].isna().sum()) == 0
    assert int(X_test["TotalCharges"].isna().sum()) == 0
    assert len(X_train) > len(X_test)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_get_preprocessor_transforms_dataframe():
    X_train, _, _, _ = get_train_test_data()
    preprocessor = get_preprocessor()
    transformed = preprocessor.fit_transform(X_train.head(20))
    assert transformed.shape[0] == 20
    assert transformed.shape[1] > len(NUMERICAL_COLS)


def test_risk_level_thresholds():
    assert _get_risk_level(0.75) == "High"
    assert _get_risk_level(0.55) == "Medium"
    assert _get_risk_level(0.20) == "Low"


def test_predict_churn_single_and_batch():
    assert get_model_path().exists()
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
        "TotalCharges": 29.85,
    }

    single_result = predict_churn(sample_customer)
    assert isinstance(single_result, dict)
    assert single_result["churn_prediction"] in (0, 1)
    assert 0.0 <= single_result["churn_probability"] <= 1.0
    assert single_result["risk_level"] in ("Low", "Medium", "High")

    batch_result = predict_churn([sample_customer, sample_customer])
    assert isinstance(batch_result, list)
    assert len(batch_result) == 2
