"""
Parte 2.2 — Predicción del Tiempo de Resolución de un Ticket (regresión, Keras)

Red neuronal con inputs mixtos (funcional API):
  - descripción del ticket -> Embedding + GlobalAveragePooling (texto)
  - categoría del ticket -> one-hot
  - prioridad -> one-hot
  - hora del día / día de la semana -> codificación cíclica (sin/cos)

Salida: tiempo estimado de resolución (horas), regresión con salida lineal.
Métricas: MAE, RMSE, R².

Salida:
  - saved_models/dl/resolution_time_model.keras
  - saved_models/dl/resolution_time_tokenizer.joblib
  - saved_models/dl/resolution_time_encoders.joblib   (categorías/prioridad -> índices)
  - saved_models/dl/resolution_time_report.json
  - saved_models/dl/resolution_time_curves.png

Ejecutar:  python dl_training/train_resolution_time_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "tickets_resolution.csv"
OUT_DIR = ROOT / "saved_models" / "dl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["TECH", "BILL", "PLAN", "CNCL", "OTHR"]
PRIORITIES = ["low", "medium", "high"]
MAX_VOCAB = 5000
MAX_LEN = 50
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


def cyclical_encode(values: np.ndarray, period: int) -> np.ndarray:
    radians = 2 * np.pi * values / period
    return np.stack([np.sin(radians), np.cos(radians)], axis=1)


def one_hot(values: pd.Series, categories: list[str]) -> np.ndarray:
    idx = values.apply(lambda v: categories.index(v) if v in categories else 0)
    return np.eye(len(categories))[idx.values]


def main() -> None:
    df = pd.read_csv(DATA_PATH).dropna(subset=["description", "category", "priority"])

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df, df["resolution_time_hours"].values, test_size=0.2, random_state=SEED)
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_df, y_train, test_size=0.15, random_state=SEED)

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_df["description"].astype(str))

    def build_inputs(sub_df: pd.DataFrame) -> dict:
        seqs = tokenizer.texts_to_sequences(sub_df["description"].astype(str))
        text_padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
        return {
            "text_input": text_padded,
            "category_input": one_hot(sub_df["category"], CATEGORIES),
            "priority_input": one_hot(sub_df["priority"], PRIORITIES),
            "hour_input": cyclical_encode(sub_df["hour_of_day"].values, 24),
            "day_input": cyclical_encode(sub_df["day_of_week"].values, 7),
        }

    train_inputs = build_inputs(X_train_df)
    val_inputs = build_inputs(X_val_df)
    test_inputs = build_inputs(X_test_df)

    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)

    # --- Arquitectura funcional con inputs mixtos ---
    text_in = layers.Input(shape=(MAX_LEN,), name="text_input")
    text_x = layers.Embedding(vocab_size, 32, mask_zero=True)(text_in)
    text_x = layers.GlobalAveragePooling1D()(text_x)
    text_x = layers.Dense(16, activation="relu")(text_x)

    category_in = layers.Input(shape=(len(CATEGORIES),), name="category_input")
    priority_in = layers.Input(shape=(len(PRIORITIES),), name="priority_input")
    hour_in = layers.Input(shape=(2,), name="hour_input")
    day_in = layers.Input(shape=(2,), name="day_input")

    merged = layers.Concatenate()([text_x, category_in, priority_in, hour_in, day_in])
    x = layers.Dense(64, activation="relu")(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(1, activation="linear", name="resolution_hours")(x)

    model = models.Model(
        inputs=[text_in, category_in, priority_in, hour_in, day_in],
        outputs=output,
        name="resolution_time_regressor",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                  tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    history = model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        epochs=60,
        batch_size=32,
        callbacks=cb_list,
        verbose=2,
    )

    y_pred = model.predict(test_inputs, verbose=0).ravel()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Test MAE={mae:.3f}h  RMSE={rmse:.3f}h  R²={r2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss (MSE)")
    axes[0].legend()
    axes[1].scatter(y_test, y_pred, alpha=0.4, s=12)
    lims = [0, max(y_test.max(), y_pred.max())]
    axes[1].plot(lims, lims, "r--", linewidth=1)
    axes[1].set_xlabel("Real (h)")
    axes[1].set_ylabel("Predicho (h)")
    axes[1].set_title(f"Real vs Predicho (R²={r2:.3f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "resolution_time_curves.png", dpi=120)
    plt.close(fig)

    model.save(OUT_DIR / "resolution_time_model.keras")
    joblib.dump(tokenizer, OUT_DIR / "resolution_time_tokenizer.joblib")
    joblib.dump({"categories": CATEGORIES, "priorities": PRIORITIES, "max_len": MAX_LEN},
                OUT_DIR / "resolution_time_encoders.joblib")

    metadata = {
        "model_version": "1.0.0",
        "mae_hours": float(mae),
        "rmse_hours": float(rmse),
        "r2": float(r2),
        "categories": CATEGORIES,
        "priorities": PRIORITIES,
        "epochs_trained": len(history.history["loss"]),
    }
    with open(OUT_DIR / "resolution_time_report.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nModelo guardado en {OUT_DIR / 'resolution_time_model.keras'}")


if __name__ == "__main__":
    main()
