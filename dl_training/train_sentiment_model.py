"""
Parte 2.1 — Red Neuronal para Clasificación de Sentimiento (TensorFlow/Keras)

Clasifica el mensaje del cliente en una interacción (customer_msg) como
positive / neutral / negative.

Arquitectura: Embedding -> LSTM -> Dense(Dropout) -> Dense(softmax)
Preprocesamiento: Tokenizer (vocab máx. 10,000) + padding (maxlen 200)
Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

Salida:
  - saved_models/dl/sentiment_model.keras
  - saved_models/dl/sentiment_tokenizer.joblib
  - saved_models/dl/sentiment_label_encoder.joblib
  - saved_models/dl/sentiment_training_curves.png
  - saved_models/dl/sentiment_confusion_matrix.png
  - saved_models/dl/sentiment_report.json

Ejecutar:  python dl_training/train_sentiment_model.py
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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "interactions.csv"
OUT_DIR = ROOT / "saved_models" / "dl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_VOCAB = 10_000
MAX_LEN = 200
EMBEDDING_DIM = 64
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


def main() -> None:
    df = pd.read_csv(DATA_PATH).dropna(subset=["customer_msg", "sentiment"])

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df["sentiment"])
    num_classes = len(label_encoder.classes_)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["customer_msg"].astype(str).tolist(), y_all, test_size=0.2,
        random_state=SEED, stratify=y_all)
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_text, y_train, test_size=0.15, random_state=SEED, stratify=y_train)

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    def to_padded(texts: list[str]) -> np.ndarray:
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    X_train = to_padded(X_train_text)
    X_val = to_padded(X_val_text)
    X_test = to_padded(X_test_text)

    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)

    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True),
        layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name="sentiment_lstm")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = OUT_DIR / "sentiment_model_best.keras"
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=cb_list,
        verbose=2,
    )

    # --- Evaluación en test ---
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    # --- Curvas de entrenamiento ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sentiment_training_curves.png", dpi=120)
    plt.close(fig)

    # --- Matriz de confusión ---
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot(
        ax=ax2, cmap="Purples", colorbar=False)
    ax2.set_title("Matriz de Confusión — Sentimiento")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "sentiment_confusion_matrix.png", dpi=120)
    plt.close(fig2)

    # --- Persistencia ---
    model.save(OUT_DIR / "sentiment_model.keras")
    joblib.dump(tokenizer, OUT_DIR / "sentiment_tokenizer.joblib")
    joblib.dump(label_encoder, OUT_DIR / "sentiment_label_encoder.joblib")

    metadata = {
        "model_version": "1.0.0",
        "classes": label_encoder.classes_.tolist(),
        "max_vocab": MAX_VOCAB,
        "max_len": MAX_LEN,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "classification_report": report,
        "epochs_trained": len(history.history["loss"]),
    }
    with open(OUT_DIR / "sentiment_report.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nModelo guardado en {OUT_DIR / 'sentiment_model.keras'}")


if __name__ == "__main__":
    main()
