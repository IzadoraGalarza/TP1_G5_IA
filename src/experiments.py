"""Protocolo comum de experimentos: mesmas dobras, mesmas métricas, uma linha por variante."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate


N_SPLITS = 5
POSITIVE_CLASS = "anomaly"

SCORING = {
    "f1_macro": "f1_macro",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_anomaly": make_scorer(f1_score, pos_label=POSITIVE_CLASS, zero_division=0),
    "precision_anomaly": make_scorer(precision_score, pos_label=POSITIVE_CLASS, zero_division=0),
    "recall_anomaly": make_scorer(recall_score, pos_label=POSITIVE_CLASS, zero_division=0),
}


def make_cv(y: pd.Series, n_splits: int = N_SPLITS) -> StratifiedKFold:
    """Dobras fixas e sem embaralhar, idênticas para todas as partes da equipe."""
    smallest_class = int(y.value_counts().min())
    if smallest_class < 2:
        raise ValueError("cada classe precisa ter pelo menos duas instâncias para validação cruzada")
    return StratifiedKFold(n_splits=min(n_splits, smallest_class), shuffle=False)


def run_experiment(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    etapa: str,
    variante: str,
    cv: StratifiedKFold | None = None,
) -> dict[str, Any]:
    """Avalia um pipeline completo por validação cruzada e devolve uma linha da tabela."""
    cv = cv if cv is not None else make_cv(y)
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1, error_score="raise")

    row: dict[str, Any] = {"etapa": etapa, "variante": variante}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        row[f"{metric}_media"] = float(values.mean())
        row[f"{metric}_dp"] = float(values.std())
    row["tempo_ajuste_s"] = float(scores["fit_time"].mean())
    return row


def results_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Tabela padronizada, ordenada pelo F1-macro médio."""
    columns = [
        "etapa",
        "variante",
        "f1_macro_media",
        "f1_macro_dp",
        "recall_anomaly_media",
        "f1_anomaly_media",
        "precision_macro_media",
        "recall_macro_media",
        "tempo_ajuste_s",
    ]
    table = pd.DataFrame(rows)
    return table[columns].sort_values("f1_macro_media", ascending=False).reset_index(drop=True)
