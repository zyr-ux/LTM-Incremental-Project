import json
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "store_item_demand.csv"
MODEL_PATH = BASE_DIR / "models" / "demand_lstm_v9.keras"
SCALER_PATH = BASE_DIR / "models" / "demand_scaler_v9.joblib"
METADATA_PATH = BASE_DIR / "models" / "demand_scaler_metadata_v9.json"
PLOT_PATH = BASE_DIR / "docs" / "demand_forecast_evaluation.png"

keras.utils.set_random_seed(42)
np.random.seed(42)

def load_and_resample_series(data_path=RAW_DATA_PATH, store_id=1, item_id=1, freq="W"):
    df = pd.read_csv(data_path)
    subset = df[(df["store"] == store_id) & (df["item"] == item_id)].copy()
    subset["date"] = pd.to_datetime(subset["date"])
    series = subset.set_index("date")["sales"].resample(freq).sum().to_frame()
    return series

def create_sliding_windows(data, window_size):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(x), np.array(y)

def build_lstm_model(window_size, n_features=1):
    model = keras.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.1),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss="mse",
        metrics=["mae"]
    )
    return model

def train_and_evaluate(window_size=12, test_weeks=12, epochs=60):
    series = load_and_resample_series()

    train_series = series.iloc[:-test_weeks]
    test_series = series.iloc[-test_weeks:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.values)
    test_scaled = scaler.transform(test_series.values)

    x_train, y_train = create_sliding_windows(train_scaled, window_size)

    test_context_scaled = np.vstack([train_scaled[-window_size:], test_scaled])
    x_test, y_test = create_sliding_windows(test_context_scaled, window_size)

    model = build_lstm_model(window_size)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )

    model.fit(
        x_train,
        y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    preds_scaled = model.predict(x_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled)
    actuals = scaler.inverse_transform(y_test)

    mae = float(np.mean(np.abs(preds - actuals)))
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mape = float(np.mean(np.abs((actuals - preds) / actuals)) * 100)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "store_id": 1,
        "item_id": 1,
        "frequency": "W",
        "window_size": window_size,
        "test_weeks": test_weeks,
        "metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2)
        },
        "recent_history": train_series.tail(window_size)["sales"].tolist()
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    plot_forecast_vs_actual(test_series.index, actuals.flatten(), preds.flatten(), PLOT_PATH)

    return model, metadata

def plot_forecast_vs_actual(dates, actuals, predicted, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actuals, marker="o", label="Actual Sales", color="#1f77b4", linewidth=2)
    plt.plot(dates, predicted, marker="s", linestyle="--", label="LSTM Forecast", color="#ff7f0e", linewidth=2)
    plt.title("RetailMind AI: LSTM Weekly Demand Forecasting vs Actuals (Store 1, Item 1)", fontsize=12, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Units Demanded / Sold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    train_and_evaluate()

