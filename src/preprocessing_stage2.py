"""Experimentos reproduzíveis de pré-processamento para a entrega de 07/11."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import CLEANING_VARIANTS, build_cleaning_pipeline


RANDOM_STATE = 42
TARGET_CANDIDATES = ("class", "classe", "label", "target", "classe_alvo", "anomalia")
KNOWN_CATEGORICAL = ("protocol_type", "service", "flag")
LEAKAGE_COLUMNS = (
    "tipo_alerta_soc",
    "score_ids_legado",
    "tempo_resposta_analista_min",
)
CONSTANT_COLUMNS = ("num_outbound_cmds",)
REDUNDANT_DERIVED_COLUMNS = ("src_bytes_kb", "dst_bytes_kb", "total_bytes")
NONNEGATIVE_COLUMNS = ("duration", "src_bytes", "dst_bytes")
RATE_COLUMNS = (
    "dst_host_srv_diff_host_rate",
    "dst_host_srv_rerror_rate",
    "rerror_rate",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
)
COUNT_LIMITS = {
    "dst_host_count": (0, 255),
    "count": (0, 511),
    "srv_count": (0, 511),
}


@dataclass(frozen=True)
class PreparationReport:
    target_column: str
    original_rows: int
    exact_duplicates_removed: int
    final_rows: int
    dropped_leakage_columns: tuple[str, ...]
    dropped_constant_columns: tuple[str, ...]
    dropped_redundant_columns: tuple[str, ...]
    invalid_values_replaced_by_nan: int
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]


def detect_target(data: pd.DataFrame, explicit_target: str | None = None) -> str:
    if explicit_target is not None:
        if explicit_target not in data.columns:
            raise ValueError(f"coluna alvo não encontrada: {explicit_target}")
        return explicit_target

    matches = [column for column in TARGET_CANDIDATES if column in data.columns]
    if not matches:
        raise ValueError(
            "coluna alvo não detectada; informe explicit_target com o nome correto"
        )
    return matches[0]


def normalize_binary_target(target: pd.Series) -> pd.Series:
    values = target.astype("string").str.strip().str.lower()
    normal_tokens = {"normal", "normal.", "0", "false", "ok"}
    anomaly_tokens = {"anomalia", "anomaly", "attack", "ataque", "1", "true", "alerta"}

    normalized = pd.Series(pd.NA, index=target.index, dtype="string")
    normalized.loc[values.isin(normal_tokens)] = "normal"
    normalized.loc[values.isin(anomaly_tokens)] = "anomaly"

    unknown = values[normalized.isna() & values.notna()].unique().tolist()
    if unknown:
        distinct = values.dropna().unique().tolist()
        normal_like = [value for value in distinct if value in normal_tokens]
        if len(distinct) == 2 and len(normal_like) == 1:
            other = next(value for value in distinct if value not in normal_tokens)
            normalized.loc[values == other] = "anomaly"
            unknown = []
    if unknown:
        raise ValueError(f"rótulos não reconhecidos: {sorted(map(str, unknown))}")
    if normalized.isna().any():
        raise ValueError("a classe possui valores ausentes")
    if set(normalized.unique()) != {"normal", "anomaly"}:
        raise ValueError("a classe precisa conter normal e anomaly")
    return normalized


def prepare_modeling_data(
    data: pd.DataFrame,
    *,
    explicit_target: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, PreparationReport]:
    """Aplica somente regras determinísticas definidas na Etapa 1.

    Estatísticas aprendidas, como medianas e categorias, permanecem no pipeline de CV.
    """
    target_column = detect_target(data, explicit_target)
    working = data.copy()
    original_rows = len(working)

    duplicate_mask = working.duplicated(keep="first")
    exact_duplicates_removed = int(duplicate_mask.sum())
    working = working.loc[~duplicate_mask].copy()
    y = normalize_binary_target(working[target_column])

    ignored_identifiers = [column for column in ("id", "ID", "Id") if column in working.columns]
    dropped_leakage = tuple(column for column in LEAKAGE_COLUMNS if column in working.columns)
    dropped_constant = tuple(column for column in CONSTANT_COLUMNS if column in working.columns)
    dropped_redundant = tuple(
        column for column in REDUNDANT_DERIVED_COLUMNS if column in working.columns
    )
    drop_columns = [
        target_column,
        *ignored_identifiers,
        *dropped_leakage,
        *dropped_constant,
        *dropped_redundant,
    ]
    X = working.drop(columns=drop_columns, errors="ignore")

    for column in KNOWN_CATEGORICAL:
        if column in X.columns:
            normalized = X[column].astype("string").str.strip().str.lower()
            # object + np.nan: o SimpleImputer não aceita o pd.NA do dtype "string".
            X[column] = normalized.astype(object).where(normalized.notna(), np.nan)

    invalid_values_replaced = 0
    for column in NONNEGATIVE_COLUMNS:
        if column in X.columns:
            numeric = pd.to_numeric(X[column], errors="coerce")
            invalid = numeric.notna() & (numeric < 0)
            invalid_values_replaced += int(invalid.sum())
            X[column] = numeric.mask(invalid)

    for column in RATE_COLUMNS:
        if column in X.columns:
            numeric = pd.to_numeric(X[column], errors="coerce")
            invalid = numeric.notna() & ~numeric.between(0, 1)
            invalid_values_replaced += int(invalid.sum())
            X[column] = numeric.mask(invalid)

    for column, (lower, upper) in COUNT_LIMITS.items():
        if column in X.columns:
            numeric = pd.to_numeric(X[column], errors="coerce")
            invalid = numeric.notna() & ~numeric.between(lower, upper)
            invalid_values_replaced += int(invalid.sum())
            X[column] = numeric.mask(invalid)

    categorical = tuple(
        column
        for column in X.columns
        if column in KNOWN_CATEGORICAL
        or pd.api.types.is_object_dtype(X[column])
        or isinstance(X[column].dtype, pd.StringDtype)
        or isinstance(X[column].dtype, pd.CategoricalDtype)
    )
    numeric = tuple(column for column in X.columns if column not in categorical)
    report = PreparationReport(
        target_column=target_column,
        original_rows=original_rows,
        exact_duplicates_removed=exact_duplicates_removed,
        final_rows=len(X),
        dropped_leakage_columns=dropped_leakage,
        dropped_constant_columns=dropped_constant,
        dropped_redundant_columns=dropped_redundant,
        invalid_values_replaced_by_nan=invalid_values_replaced,
        categorical_columns=categorical,
        numeric_columns=numeric,
    )
    return X, y, report


def build_preprocessor(report: PreparationReport) -> ColumnTransformer:
    categorical_pipeline = SklearnPipeline(
        steps=[
            ("imputacao", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    numeric_pipeline = SklearnPipeline(
        steps=[
            ("imputacao", SimpleImputer(strategy="median", add_indicator=True)),
            ("padronizacao", StandardScaler(with_mean=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categoricos", categorical_pipeline, list(report.categorical_columns)),
            ("numericos", numeric_pipeline, list(report.numeric_columns)),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def build_experiment_pipeline(
    report: PreparationReport,
    sampler: Any,
    *,
    n_components: int = 20,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessamento", build_preprocessor(report)),
            (
                "reducao_dimensionalidade",
                TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE),
            ),
            ("balanceamento", sampler),
            (
                "classificador_referencia",
                LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
            ),
        ]
    )


def make_duplicate_groups(X: pd.DataFrame) -> pd.Series:
    """Mantém assinaturas idênticas na mesma dobra para reduzir contaminação."""
    normalized = X.astype("string").fillna("<NA>")
    return pd.util.hash_pandas_object(normalized, index=False).astype("string")


def _f1_macro(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    return float(f1_score(y, estimator.predict(X), average="macro", zero_division=0))


def _precision_macro(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    return float(
        precision_score(y, estimator.predict(X), average="macro", zero_division=0)
    )


def _recall_macro(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    return float(recall_score(y, estimator.predict(X), average="macro", zero_division=0))


def _precision_anomaly(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    return float(
        precision_score(y, estimator.predict(X), pos_label="anomaly", zero_division=0)
    )


def _recall_anomaly(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
    return float(recall_score(y, estimator.predict(X), pos_label="anomaly", zero_division=0))


def _scoring() -> dict[str, Any]:
    return {
        "f1_macro": _f1_macro,
        "precision_macro": _precision_macro,
        "recall_macro": _recall_macro,
        "precision_anomaly": _precision_anomaly,
        "recall_anomaly": _recall_anomaly,
    }


def evaluate_cleaning_strategies(
    data: pd.DataFrame,
    *,
    explicit_target: str | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compara as decisões de redundância/inconsistência uma por vez."""
    target_column = detect_target(data, explicit_target)
    y = normalize_binary_target(data[target_column])
    identifiers = [column for column in ("id", "ID", "Id") if column in data.columns]
    X = data.drop(columns=[target_column, *identifiers], errors="ignore")
    groups = make_duplicate_groups(X)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    rows: list[dict[str, Any]] = []

    for variant_name, config in CLEANING_VARIANTS.items():
        scores = cross_validate(
            build_cleaning_pipeline(config),
            X,
            y,
            groups=groups,
            cv=cv,
            scoring=_scoring(),
            n_jobs=1,
            error_score="raise",
        )
        rows.append(
            {
                "variante": variant_name,
                "f1_macro_media": float(np.mean(scores["test_f1_macro"])),
                "f1_macro_desvio": float(np.std(scores["test_f1_macro"], ddof=1)),
                "precision_anomaly_media": float(
                    np.mean(scores["test_precision_anomaly"])
                ),
                "recall_anomaly_media": float(np.mean(scores["test_recall_anomaly"])),
                "tempo_ajuste_s_media": float(np.mean(scores["fit_time"])),
            }
        )
    return pd.DataFrame(rows).sort_values("f1_macro_media", ascending=False).reset_index(drop=True)


def evaluate_preprocessing_strategies(
    X: pd.DataFrame,
    y: pd.Series,
    report: PreparationReport,
    *,
    n_splits: int = 5,
    n_components: int = 20,
) -> pd.DataFrame:
    strategies = {
        "sem_balanceamento": "passthrough",
        "random_under_sampler": RandomUnderSampler(random_state=RANDOM_STATE),
        "random_over_sampler": RandomOverSampler(random_state=RANDOM_STATE),
    }
    scoring = _scoring()
    groups = make_duplicate_groups(X)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    rows: list[dict[str, Any]] = []

    for strategy_name, sampler in strategies.items():
        pipeline = build_experiment_pipeline(
            report,
            sampler,
            n_components=n_components,
        )
        scores = cross_validate(
            pipeline,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        row: dict[str, Any] = {
            "estrategia": strategy_name,
            "codificacao": "one_hot",
            "transformacao_numerica": "standard_scaler",
            "reducao_dimensionalidade": f"truncated_svd_{n_components}",
            "classificador_referencia": "regressao_logistica",
            "dobras": n_splits,
        }
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_media"] = float(np.mean(values))
            row[f"{metric}_desvio"] = float(np.std(values, ddof=1))
        row["tempo_ajuste_s_media"] = float(np.mean(scores["fit_time"]))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("f1_macro_media", ascending=False).reset_index(drop=True)


def report_as_dict(report: PreparationReport) -> dict[str, Any]:
    return asdict(report)
