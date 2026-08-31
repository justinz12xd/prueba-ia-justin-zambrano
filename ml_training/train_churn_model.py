"""
Parte 1.2 — Modelo de Predicción de Churn (scikit-learn)

Entrena un modelo de clasificación binaria (churn_status) con probabilidades,
a partir de features del cliente. Incluye:
  - EDA básico (distribución de churn, correlaciones)
  - Manejo de nulos y de desbalanceo de clases
  - Feature engineering (>= 2 features derivados)
  - Métricas: AUC-ROC, curva precision-recall
  - Explicabilidad: feature importance

Salida:
  - saved_models/ml/churn_model.joblib        (pipeline: preprocesador + modelo)
  - saved_models/ml/churn_report.json
  - saved_models/ml/churn_roc_pr_curves.png
  - saved_models/ml/churn_feature_importance.png

Ejecutar:  python ml_training/train_churn_model.py
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (PrecisionRecallDisplay, RocCurveDisplay, average_precision_score,
                              roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "customers.csv"
OUT_DIR = ROOT / "saved_models" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "tenure_months", "monthly_charge", "total_charges", "num_tickets",
    "avg_satisfaction", "charge_per_tenure", "tickets_per_tenure",
]
CATEGORICAL_FEATURES = ["contract_type", "payment_method"]
TARGET = "churn_status"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features derivados a partir de las columnas base."""
    df = df.copy()
    # 1) Cargo promedio por mes de antigüedad (detecta clientes "caros" relativos)
    df["charge_per_tenure"] = df["total_charges"] / df["tenure_months"].replace(0, 1)
    # 2) Densidad de tickets por antigüedad (clientes con muchos problemas recientes)
    df["tickets_per_tenure"] = df["num_tickets"] / (df["tenure_months"].replace(0, 1) / 12)
    return df


def run_eda(df: pd.DataFrame) -> dict:
    churn_dist = df[TARGET].value_counts(normalize=True).to_dict()
    numeric_corr = df[NUMERIC_FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
    return {
        "churn_distribution": {str(k): float(v) for k, v in churn_dist.items()},
        "correlation_with_churn": numeric_corr.round(3).to_dict(),
        "null_counts_before_imputation": df.isna().sum().to_dict(),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["avg_satisfaction"] = pd.to_numeric(df["avg_satisfaction"], errors="coerce")
    df = engineer_features(df)

    eda = run_eda(df)
    print("=== EDA ===")
    print("Distribución churn:", eda["churn_distribution"])
    print("Correlación con churn:\n", pd.Series(eda["correlation_with_churn"]))

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])

    # class_weight="balanced" maneja el desbalanceo sin necesidad de resampling externo (SMOTE)
    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    }

    results = {}
    fitted = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)
        results[name] = {"auc_roc": float(auc), "average_precision": float(ap)}
        fitted[name] = pipe
        print(f"{name}: AUC-ROC={auc:.4f}  AP(PR)={ap:.4f}")

    best_name = max(results, key=lambda k: results[k]["auc_roc"])
    best_pipe = fitted[best_name]
    print(f"\n>> Mejor modelo: {best_name}")

    proba_best = best_pipe.predict_proba(X_test)[:, 1]

    # Curvas ROC y Precision-Recall
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba_best, ax=axes[0])
    axes[0].set_title("Curva ROC — Churn")
    PrecisionRecallDisplay.from_predictions(y_test, proba_best, ax=axes[1])
    axes[1].set_title("Curva Precision-Recall — Churn")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "churn_roc_pr_curves.png", dpi=120)
    plt.close(fig)

    # Feature importance (explicabilidad)
    feature_names = (
        NUMERIC_FEATURES +
        list(best_pipe.named_steps["preprocess"]
             .named_transformers_["cat"].named_steps["onehot"]
             .get_feature_names_out(CATEGORICAL_FEATURES))
    )
    importances = best_pipe.named_steps["clf"].feature_importances_
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    imp_series.plot(kind="barh", ax=ax2, color="#3B82F6")
    ax2.set_title(f"Feature Importance — {best_name}")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "churn_feature_importance.png", dpi=120)
    plt.close(fig2)

    joblib.dump(best_pipe, OUT_DIR / "churn_model.joblib")

    metadata = {
        "best_model": best_name,
        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,
        "model_version": "1.0.0",
        "eda": eda,
        "results": results,
        "feature_importance": imp_series.sort_values(ascending=False).round(4).to_dict(),
    }
    with open(OUT_DIR / "churn_report.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nModelo guardado en {OUT_DIR / 'churn_model.joblib'}")
    print(f"Reporte guardado en {OUT_DIR / 'churn_report.json'}")


if __name__ == "__main__":
    main()
