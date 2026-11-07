"""Baseline sem balanceamento, com todo pré-processamento dentro do pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.data_validation import CATEGORICAL_COLUMNS, ID_COLUMN, TARGET_COLUMN


RANDOM_STATE = 42


def split_features_target(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in train.columns:
        raise ValueError(f"coluna alvo ausente: {TARGET_COLUMN}")

    columns_to_drop = [TARGET_COLUMN]
    if ID_COLUMN in train.columns:
        columns_to_drop.append(ID_COLUMN)

    X = train.drop(columns=columns_to_drop)
    y = train[TARGET_COLUMN].astype(str).str.lower()
    return X, y


def build_baseline_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical = [column for column in CATEGORICAL_COLUMNS if column in X.columns]
    numeric = [column for column in X.columns if column not in categorical]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical),
            ("numeric", numeric_pipeline, numeric),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )


def evaluate_baseline(
    train: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Avalia o baseline em dobras fixas e retorna métricas e relatório fora da amostra."""
    X, y = split_features_target(train)
    smallest_class = int(y.value_counts().min())
    if smallest_class < 2:
        raise ValueError("cada classe precisa ter pelo menos duas instâncias para validação cruzada")
    effective_splits = min(n_splits, smallest_class)

    cv = StratifiedKFold(n_splits=effective_splits, shuffle=False)
    pipeline = build_baseline_pipeline(X)
    scoring = {
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    predictions = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=1)

    fold_results = pd.DataFrame(
        {
            "dobra": range(1, effective_splits + 1),
            "f1_macro": scores["test_f1_macro"],
            "precision_macro": scores["test_precision_macro"],
            "recall_macro": scores["test_recall_macro"],
            "tempo_ajuste_s": scores["fit_time"],
        }
    )
    report = classification_report(y, predictions, output_dict=True, zero_division=0)
    return fold_results, report
