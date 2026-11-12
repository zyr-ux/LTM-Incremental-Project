# Day 10 DevLog: Seq2Seq & Attention Demand Forecasting (Multi-Horizon Procurement Optimization)

**Date**: 2026-11-12  
**Platform**: RetailMind AI  
**Increment**: 10 / 35  
**Deliverable**: Multi-Horizon Sequence-to-Sequence (Seq2Seq) with Bahdanau Additive Attention Forecasting Engine & Dynamic Procurement Planning Service  
**Focus**: Multi-step sliding windows (12 look-back weeks $\rightarrow$ 4 forecast horizon weeks), Encoder-Decoder recurrent modeling, Bahdanau Attention Layer, Attention weights heatmaps, Multi-horizon procurement recommendations  
**Stack**: `tensorflow`, `keras`, `pandas`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`  

---

## 1. Objectives & Retail Context

Accurate multi-week demand forecasts allow retail inventory and procurement managers to place supplier purchase orders well in advance, avoiding expedited freight charges while minimizing stockout risks:
1. **Multi-Horizon Framing**:
   - Single-step models (Day 9) require recursive, autoregressive rollout for multi-week planning, compounding errors at every step.
   - Day 10 shifts to a Sequence-to-Sequence (Seq2Seq) structure where the model maps an input history window ($W_{in} = 12$ weeks) directly into a vector of multi-horizon future demands ($H_{out} = 4$ weeks).
2. **Bahdanau Additive Attention Mechanism**:
   - Integrates an additive attention layer that computes a alignment score between the current decoder hidden state and all encoder hidden time-states.
   - Allows the decoder to focus selectively on relevant historical weeks (such as recent inflection points or seasonal cycles) rather than relying on a single static context bottleneck vector.
3. **Attention Explainability**:
   - Generates an interpretable $4 \times 12$ attention weight matrix for every forecast window, revealing which historical weeks drove each horizon's prediction.
4. **Supply Chain Procurement Service**:
   - Enhances `src/demand_service.py` with `predict_multi_step_demand` to emit a 4-week demand projection with safety-stock buffered reorder recommendations for warehouse procurement.

---

## 2. Methodology & Architecture

### A. Window Framing (`src/demand_forecast_seq2seq.py`)
- **Dataset**: Store 1, Item 1 weekly resampled sales (261 weekly demand observations).
- **Split Strategy**:
  - Training Set: First 249 weeks (95.4%).
  - Holdout Test Set: Final 12 weeks (4.6%).
- **Window Reshaping**:
  - Input history sequence: $X \in \mathbb{R}^{B \times 12 \times 1}$ (past 12 weeks).
  - Target output sequence: $Y \in \mathbb{R}^{B \times 4 \times 1}$ (future 4 weeks).

### B. Seq2Seq with Bahdanau Attention Architecture
- **Encoder**:
  - `Input(shape=(12, 1))`
  - `LSTM(64, return_sequences=True, return_state=True)`: Emits all hidden states $h_1, \dots, h_{12}$ and the final recurrent cell state $(h_{12}, c_{12})$.
- **Bahdanau Attention Layer**:
  - Additive score: $e_{t, i} = v^T \tanh(W_1 s_t + W_2 h_i)$.
  - Attention weights: $\alpha_{t} = \text{softmax}(e_{t})$.
  - Context vector: $c_t = \sum_{i=1}^{12} \alpha_{t, i} h_i$.
- **Decoder**:
  - `LSTMCell(64)`: Initialized with encoder final state. At each step $t \in \{1, 2, 3, 4\}$, concatenates previous step output with $c_t$.
  - Projection: `Dense(1)` linear output neuron.
- **Optimization**:
  - Loss: `MSE`, Optimizer: `Adam(learning_rate=0.005)`, `EarlyStopping(patience=15)`.

---

## 3. Holdout Evaluation & Single-Step vs Multi-Step Comparison

Evaluated on the 12-week holdout test horizon:

| Model Architecture | Horizon Target | Overall MAE | Overall RMSE | Overall MAPE | Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Day 9 LSTM (Single-Step)** | $t+1$ (next week) | **18.01** | **20.88** | **13.77%** | Single-step direct regression |
| **Day 10 Seq2Seq + Attention** | $t+1$ (step 1) | **14.08** | — | — | **+21.8% error reduction on step 1** |
| **Day 10 Seq2Seq + Attention** | $t+1 \dots t+4$ (4 weeks) | **21.45** | **27.89** | **16.97%** | Direct multi-horizon forecast across a full month |

### Step-by-Step Breakdown Across the 4-Week Horizon:
```text
Forecast Horizon   Lead Time    MAE (Units)   Interpretation
-------------------------------------------------------------------------
Step 1 (t+1)       1 Week             14.08   Immediate replenishment
Step 2 (t+2)       2 Weeks            18.62   Short-term procurement
Step 3 (t+3)       3 Weeks            25.22   Medium-term supplier order
Step 4 (t+4)       4 Weeks            27.87   Monthly procurement lock
```

- Evaluation chart saved to: `docs/demand_seq2seq_evaluation.png`
- Attention weights heatmap saved to: `docs/demand_attention_weights.png`

---

## 4. Production Demand & Procurement Service (`src/demand_service.py`)

The upgraded demand service enables multi-week procurement planning with attention inspection:

```python
from src.demand_service import predict_multi_step_demand

# Emits 4-week demand projection with safety-stock reorder targets
result = predict_multi_step_demand(return_attention=True)
# Returns:
# {
#   "store_id": 1,
#   "item_id": 1,
#   "frequency": "W",
#   "forecast_horizon_weeks": 4,
#   "forecasted_demand": [154.3, 156.1, 153.8, 149.9],
#   "inventory_reorder_recommendation": [178, 180, 177, 173],
#   "safety_stock_buffer_percent": 15,
#   "model_architecture": "Seq2Seq_LSTM_Bahdanau_Attention",
#   "attention_weights": [[...], [...], [...], [...]],
#   "model_metrics": {...}
# }
```

