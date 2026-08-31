"""
Parte 1.1 — Clasificador de Tickets de Soporte (scikit-learn)

Entrena y compara 2 pipelines (TF-IDF + Logistic Regression y TF-IDF + LinearSVC)
para clasificar la descripción de un ticket en una de las categorías:
TECH, BILL, PLAN, CNCL, OTHR.

Salida:
  - saved_models/ml/ticket_classifier.joblib   (mejor pipeline, listo para inferencia)
  - saved_models/ml/ticket_classifier_report.json (métricas de ambos modelos)
  - saved_models/ml/confusion_matrix_tickets.png

Ejecutar:  python ml_training/train_ticket_classifier.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.ml_runtime.text_preprocessing import (  # noqa: E402  (ver nota abajo)
    SPANISH_STOPWORDS, normalize_text, validate_min_length as validate_input_text)

DATA_PATH = ROOT / "data" / "tickets_train.csv"
OUT_DIR = ROOT / "saved_models" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["TECH", "BILL", "PLAN", "CNCL", "OTHR"]

# NOTA: `normalize_text` se importa desde app.ml_runtime.text_preprocessing (en vez de
# definirse aquí) a propósito: el TfidfVectorizer serializa esta función con joblib, y
# joblib necesita poder reimportarla por su ruta de módulo real al cargar el modelo
# desde otro proceso (la API). Si viviera en este script, quedaría pickleada como
# `__main__.normalize_text` y fallaría al cargar fuera de este archivo.


def build_pipelines() -> dict[str, Pipeline]:
    vectorizer_kwargs = dict(
        preprocessor=normalize_text,
        stop_words=SPANISH_STOPWORDS,
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        sublinear_tf=True,
    )
    logreg_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**vectorizer_kwargs)),
        ("clf", LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced")),
    ])

    # LinearSVC no tiene predict_proba nativo -> se calibra para exponer probabilidades
    svc_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**vectorizer_kwargs)),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced"), method="sigmoid", cv=3)),
    ])

    return {"logistic_regression": logreg_pipeline, "linear_svc_calibrated": svc_pipeline}


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["description", "category"])
    df = df[df["description"].str.len() >= 10]

    X, y = df["description"], df["category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = build_pipelines()

    results = {}
    for name, pipe in pipelines.items():
        print(f"\n=== Entrenando: {name} ===")
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        report = classification_report(y_test, y_pred, labels=CATEGORIES,
                                        output_dict=True, zero_division=0)
        test_f1_macro = f1_score(y_test, y_pred, average="macro")

        results[name] = {
            "cv_f1_macro_5fold": {
                "mean": float(cv_scores.mean()),
                "std": float(cv_scores.std()),
                "folds": cv_scores.tolist(),
            },
            "test_accuracy": report["accuracy"],
            "test_f1_macro": float(test_f1_macro),
            "classification_report": report,
        }
        print(f"CV F1-macro (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Test accuracy: {report['accuracy']:.4f} | Test F1-macro: {test_f1_macro:.4f}")
        print(classification_report(y_test, y_pred, labels=CATEGORIES, zero_division=0))

    # Selecciona el mejor modelo por F1-macro en test
    best_name = max(results, key=lambda k: results[k]["test_f1_macro"])
    best_pipeline = pipelines[best_name]
    print(f"\n>> Mejor modelo: {best_name}")

    # Matriz de confusión del mejor modelo
    y_pred_best = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=CATEGORIES)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=CATEGORIES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de Confusión — {best_name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix_tickets.png", dpi=120)
    plt.close(fig)

    # Persistencia
    joblib.dump(best_pipeline, OUT_DIR / "ticket_classifier.joblib")
    metadata = {
        "best_model": best_name,
        "categories": CATEGORIES,
        "n_samples_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_version": "1.0.0",
        "results": results,
    }
    with open(OUT_DIR / "ticket_classifier_report.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nModelo guardado en {OUT_DIR / 'ticket_classifier.joblib'}")
    print(f"Reporte guardado en {OUT_DIR / 'ticket_classifier_report.json'}")


if __name__ == "__main__":
    main()
