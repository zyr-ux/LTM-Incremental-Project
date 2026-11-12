import json
import joblib
from pathlib import Path
import numpy as np
import keras
from src.demand_forecast_seq2seq import Seq2SeqAttentionModel, BahdanauAttention

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "demand_seq2seq_attention_v10.keras"
SCALER_PATH = BASE_DIR / "models" / "demand_scaler_v10.joblib"
METADATA_PATH = BASE_DIR / "models" / "demand_seq2seq_metadata_v10.json"

_model = None
_scaler = None
_metadata = None

def _load_artifacts():
    global _model, _scaler, _metadata
    if _model is None:
        _model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "Seq2SeqAttentionModel": Seq2SeqAttentionModel,
                "BahdanauAttention": BahdanauAttention
            }
        )
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    if _metadata is None:
        with open(METADATA_PATH, "r") as f:
            _metadata = json.load(f)
    return _model, _scaler, _metadata

def predict_multi_step_demand(recent_sales_window=None, return_attention=False):
    model, scaler, metadata = _load_artifacts()
    input_steps = metadata["input_steps"]
    forecast_horizon = metadata["forecast_horizon"]

    if recent_sales_window is None:
        recent_sales_window = metadata.get("recent_history", [])[-input_steps:]

    if len(recent_sales_window) < input_steps:
        raise ValueError(f"Expected at least {input_steps} past periods, got {len(recent_sales_window)}")

    current_window = np.array(recent_sales_window[-input_steps:], dtype=np.float32).reshape(-1, 1)
    scaled_window = scaler.transform(current_window).flatten()
    x_input = scaled_window.reshape(1, input_steps, 1)

    preds_scaled, att_weights = model.predict(x_input, verbose=0)
    preds_scaled_2d = preds_scaled[0].reshape(-1, 1)
    forecasts = scaler.inverse_transform(preds_scaled_2d).flatten()
    forecast_list = [round(max(0.0, float(val)), 1) for val in forecasts]

    recommended_reorder = [int(np.ceil(val * 1.15)) for val in forecast_list]

    response = {
        "store_id": metadata["store_id"],
        "item_id": metadata["item_id"],
        "frequency": metadata["frequency"],
        "forecast_horizon_weeks": forecast_horizon,
        "forecasted_demand": forecast_list,
        "inventory_reorder_recommendation": recommended_reorder,
        "safety_stock_buffer_percent": 15,
        "model_architecture": "Seq2Seq_LSTM_Bahdanau_Attention",
        "model_metrics": metadata["metrics"]
    }

    if return_attention:
        response["attention_weights"] = np.round(att_weights[0], 4).tolist()

    return response

def predict_demand(recent_sales_window=None, steps_ahead=1):
    result = predict_multi_step_demand(recent_sales_window=recent_sales_window)
    forecasts = result["forecasted_demand"]
    reorders = result["inventory_reorder_recommendation"]

    if steps_ahead == 1:
        result["forecasted_demand"] = forecasts[0]
        result["inventory_reorder_recommendation"] = reorders[0]
    else:
        result["forecasted_demand"] = forecasts[:steps_ahead]
        result["inventory_reorder_recommendation"] = reorders[:steps_ahead]
    return result
