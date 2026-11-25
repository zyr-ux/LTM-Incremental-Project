import json
from pathlib import Path
import joblib
import numpy as np
import pytest


class _FakeSeq2SeqModel:
    def predict(self, x_input, verbose=0):
        batch = x_input.shape[0]
        preds = np.full((batch, 4), 0.5, dtype=np.float32)
        att = np.full((batch, 4, x_input.shape[1]), 0.1, dtype=np.float32)
        return preds, att


def test_demand_service_multi_step_and_single_step(monkeypatch):
    import sys
    import types

    dummy_seq2seq = types.ModuleType("src.demand_forecast_seq2seq")
    setattr(dummy_seq2seq, "Seq2SeqAttentionModel", object)
    setattr(dummy_seq2seq, "BahdanauAttention", object)
    monkeypatch.setitem(sys.modules, "src.demand_forecast_seq2seq", dummy_seq2seq)
    monkeypatch.setitem(sys.modules, "keras", types.ModuleType("keras"))

    import src.demand_service as ds

    scaler = joblib.load(Path("models/demand_scaler_v10.joblib"))
    with open(Path("models/demand_seq2seq_metadata_v10.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    monkeypatch.setattr(ds, "_load_artifacts", lambda: (_FakeSeq2SeqModel(), scaler, metadata))

    window = [20.0] * metadata["input_steps"]
    multi = ds.predict_multi_step_demand(recent_sales_window=window, return_attention=True)
    assert multi["forecast_horizon_weeks"] == 4
    assert len(multi["forecasted_demand"]) == 4
    assert len(multi["inventory_reorder_recommendation"]) == 4
    assert "attention_weights" in multi

    single = ds.predict_demand(recent_sales_window=window, steps_ahead=1)
    assert isinstance(single["forecasted_demand"], float)
    assert isinstance(single["inventory_reorder_recommendation"], int)


def test_demand_service_short_window_raises(monkeypatch):
    import sys
    import types

    dummy_seq2seq = types.ModuleType("src.demand_forecast_seq2seq")
    setattr(dummy_seq2seq, "Seq2SeqAttentionModel", object)
    setattr(dummy_seq2seq, "BahdanauAttention", object)
    monkeypatch.setitem(sys.modules, "src.demand_forecast_seq2seq", dummy_seq2seq)
    monkeypatch.setitem(sys.modules, "keras", types.ModuleType("keras"))

    import src.demand_service as ds

    scaler = joblib.load(Path("models/demand_scaler_v10.joblib"))
    with open(Path("models/demand_seq2seq_metadata_v10.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    monkeypatch.setattr(ds, "_load_artifacts", lambda: (_FakeSeq2SeqModel(), scaler, metadata))

    with pytest.raises(ValueError):
        ds.predict_multi_step_demand(recent_sales_window=[10.0, 12.0])
