import json
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras import layers

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "store_item_demand.csv"
MODEL_PATH = BASE_DIR / "models" / "demand_seq2seq_attention_v10.keras"
SCALER_PATH = BASE_DIR / "models" / "demand_scaler_v10.joblib"
METADATA_PATH = BASE_DIR / "models" / "demand_seq2seq_metadata_v10.json"
FORECAST_PLOT_PATH = BASE_DIR / "docs" / "demand_seq2seq_evaluation.png"
ATTENTION_PLOT_PATH = BASE_DIR / "docs" / "demand_attention_weights.png"

keras.utils.set_random_seed(42)
np.random.seed(42)

def load_and_resample_series(data_path=RAW_DATA_PATH, store_id=1, item_id=1, freq="W"):
    df = pd.read_csv(data_path)
    subset = df[(df["store"] == store_id) & (df["item"] == item_id)].copy()
    subset["date"] = pd.to_datetime(subset["date"])
    series = subset.set_index("date")["sales"].resample(freq).sum().to_frame()
    return series

def create_seq2seq_windows(data, input_steps=12, output_steps=4):
    x, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        x.append(data[i : i + input_steps])
        y.append(data[i + input_steps : i + input_steps + output_steps])
    return np.array(x), np.array(y)

@keras.saving.register_keras_serializable(package="retailmind")
class BahdanauAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

@keras.saving.register_keras_serializable(package="retailmind")
class Seq2SeqAttentionModel(keras.Model):
    def __init__(self, enc_units=64, dec_units=64, forecast_horizon=4, **kwargs):
        super().__init__(**kwargs)
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.forecast_horizon = forecast_horizon
        self.encoder_lstm = layers.LSTM(enc_units, return_sequences=True, return_state=True)
        self.decoder_cell = layers.LSTMCell(dec_units)
        self.attention = BahdanauAttention(dec_units)
        self.dense_out = layers.Dense(1)

    def call(self, inputs, training=None):
        enc_seq, state_h, state_c = self.encoder_lstm(inputs)
        dec_state = [state_h, state_c]
        curr_input = inputs[:, -1, :]

        outputs = []
        attentions = []
        for _ in range(self.forecast_horizon):
            context_vec, att_weights = self.attention(dec_state[0], enc_seq)
            cell_input = tf.concat([curr_input, context_vec], axis=-1)
            cell_out, dec_state = self.decoder_cell(cell_input, dec_state)
            step_pred = self.dense_out(cell_out)
            outputs.append(step_pred)
            attentions.append(tf.squeeze(att_weights, axis=-1))
            curr_input = step_pred

        pred_seq = tf.stack(outputs, axis=1)
        att_seq = tf.stack(attentions, axis=1)
        return pred_seq, att_seq

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, _ = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred, _ = self(x, training=False)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update({
            "enc_units": self.enc_units,
            "dec_units": self.dec_units,
            "forecast_horizon": self.forecast_horizon
        })
        return config

def plot_forecast_horizons(dates, actual_full, seq2seq_preds, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual_full, marker="o", label="Actual Sales", color="#1f77b4", linewidth=2)
    pred_steps = len(seq2seq_preds)
    pred_dates = dates[-pred_steps:]
    plt.plot(pred_dates, seq2seq_preds, marker="s", linestyle="--", label="Seq2Seq Attention Forecast", color="#2ca02c", linewidth=2)
    plt.title("RetailMind AI: Seq2Seq Attention Demand Forecasting (Store 1, Item 1)", fontsize=12, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Units Demanded / Sold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_attention_matrix(attention_weights, input_steps, output_steps, output_path):
    plt.figure(figsize=(9, 4.5))
    x_labels = [f"t-{input_steps - i}" for i in range(input_steps)]
    y_labels = [f"t+{j + 1}" for j in range(output_steps)]
    sns.heatmap(attention_weights, annot=True, fmt=".2f", cmap="Blues", xticklabels=x_labels, yticklabels=y_labels)
    plt.title("RetailMind AI: Seq2Seq Attention Weights Matrix (Look-Back Focus)", fontsize=12, fontweight="bold")
    plt.xlabel("Input History Week")
    plt.ylabel("Forecast Horizon Step")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def train_and_evaluate(input_steps=12, forecast_horizon=4, test_weeks=12, epochs=75):
    series = load_and_resample_series()

    train_series = series.iloc[:-test_weeks]
    test_series = series.iloc[-test_weeks:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.values)
    test_scaled = scaler.transform(test_series.values)

    x_train, y_train = create_seq2seq_windows(train_scaled, input_steps, forecast_horizon)

    full_context_scaled = np.vstack([train_scaled[-input_steps:], test_scaled])
    x_test, y_test = create_seq2seq_windows(full_context_scaled, input_steps, forecast_horizon)

    model = Seq2SeqAttentionModel(enc_units=64, dec_units=64, forecast_horizon=forecast_horizon)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss="mse", metrics=["mae"])

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    model.fit(
        x_train,
        y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    preds_scaled, att_weights_all = model.predict(x_test, verbose=0)

    batch_size = preds_scaled.shape[0]
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(batch_size, forecast_horizon)
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(batch_size, forecast_horizon)

    mae_by_horizon = [float(np.mean(np.abs(preds[:, h] - actuals[:, h]))) for h in range(forecast_horizon)]
    overall_mae = float(np.mean(np.abs(preds - actuals)))
    overall_rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    overall_mape = float(np.mean(np.abs((actuals - preds) / actuals)) * 100)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "store_id": 1,
        "item_id": 1,
        "frequency": "W",
        "input_steps": input_steps,
        "forecast_horizon": forecast_horizon,
        "test_weeks": test_weeks,
        "metrics": {
            "overall_mae": round(overall_mae, 2),
            "overall_rmse": round(overall_rmse, 2),
            "overall_mape": round(overall_mape, 2),
            "mae_per_step": [round(m, 2) for m in mae_by_horizon]
        },
        "recent_history": train_series.tail(input_steps)["sales"].tolist()
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    latest_pred = preds[-1]
    plot_forecast_horizons(test_series.index, test_series["sales"].values, latest_pred, FORECAST_PLOT_PATH)

    latest_att = att_weights_all[-1]
    plot_attention_matrix(latest_att, input_steps, forecast_horizon, ATTENTION_PLOT_PATH)

    print("Seq2Seq Attention Training Finished!")
    print(f"Overall MAE: {overall_mae:.2f}, RMSE: {overall_rmse:.2f}, MAPE: {overall_mape:.2f}%")
    print(f"Horizon MAEs (Weeks t+1 to t+{forecast_horizon}): {[round(m, 2) for m in mae_by_horizon]}")

    return model, metadata

if __name__ == "__main__":
    train_and_evaluate()
