import json
import joblib
from pathlib import Path
import numpy as np
import keras

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "demand_lstm_v9.keras"
SCALER_PATH = BASE_DIR / "models" / "demand_scaler_v9.joblib"
METADATA_PATH = BASE_DIR / "models" / "demand_scaler_metadata_v9.json"

_model = None
_scaler = None
_metadata = None

def _load_artifacts():
    global _model, _scaler, _metadata
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    if _metadata is None:
        with open(METADATA_PATH, "r") as f:
            _metadata = json.load(f)
    return _model, _scaler, _metadata

def predict_demand(recent_sales_window=None, steps_ahead=1):
    model, scaler, metadata = _load_artifacts()
    window_size = metadata["window_size"]

    if recent_sales_window is None:
        recent_sales_window = metadata.get("recent_history", [])[-window_size:]

    if len(recent_sales_window) < window_size:
        raise ValueError(f"Expected at least {window_size} past periods, got {len(recent_sales_window)}")

    current_window = list(recent_sales_window[-window_size:])
    forecasts = []

    for _ in range(steps_ahead):
        scaled_window = scaler.transform(np.array(current_window).reshape(-1, 1)).flatten()
        x_input = scaled_window.astype(np.float32).reshape(1, window_size, 1)

        pred_scaled = float(model.predict(x_input, verbose=0)[0, 0])
        pred_sales = max(0.0, float(scaler.inverse_transform([[pred_scaled]])[0, 0]))

        forecasts.append(round(pred_sales, 1))
        current_window.append(pred_sales)
        current_window.pop(0)

    recommended_reorder = [int(np.ceil(f * 1.15)) for f in forecasts]

    return {
        "store_id": metadata["store_id"],
        "item_id": metadata["item_id"],
        "frequency": metadata["frequency"],
        "forecasted_demand": forecasts if steps_ahead > 1 else forecasts[0],
        "inventory_reorder_recommendation": recommended_reorder if steps_ahead > 1 else recommended_reorder[0],
        "safety_stock_buffer_percent": 15,
        "model_metrics": metadata["metrics"]
    }
