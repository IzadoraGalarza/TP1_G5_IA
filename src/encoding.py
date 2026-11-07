"""Parte 2 da Etapa 2: imputação categórica e conversão simbólico-numérica.

Cada variante muda só a imputação ou a codificação dos atributos categóricos; o resto segue a
configuração de controle de ``docs/etapa2_divisao.md`` (mediana + indicador, StandardScaler,
Regressão Logística, sem balanceamento e sem redução). Tudo é ajustado dentro da dobra de treino.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.experiments import SCORING
from src.preprocessing_stage2 import PreparationReport, make_duplicate_groups


RANDOM_STATE = 42
MISSING_TOKEN = "ausente"
HIGH_CARDINALITY_COLUMN = "service"
# Categorias abaixo de 0,1% do treino da dobra viram um único nível "infrequente".
RARE_MIN_FREQUENCY = 0.001


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Troca cada categoria pela sua frequência relativa no treino; categoria nova vira 0."""

    def fit(self, X: Any, y: Any = None) -> "FrequencyEncoder":
        frame = pd.DataFrame(X)
        self.frequencies_ = [frame[column].value_counts(normalize=True) for column in frame.columns]
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = pd.DataFrame(X)
        encoded = [
            frame[column].map(frequencies).astype(float).fillna(0.0).to_numpy()
            for column, frequencies in zip(frame.columns, self.frequencies_)
        ]
        return np.column_stack(encoded)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names = input_features if input_features is not None else range(self.n_features_in_)
        return np.asarray([f"freq_{name}" for name in names], dtype=object)


@dataclass(frozen=True)
class EncodingVariant:
    nome: str
    imputacao: str  # "moda" | "ausente"
    codificacao: str  # "one_hot" | "one_hot_raras" | "ordinal" | "frequencia" | "hibrida"


ENCODING_VARIANTS = (
    EncodingVariant("E0 controle: moda + one-hot", "moda", "one_hot"),
    EncodingVariant("E1 categoria 'ausente' + one-hot", "ausente", "one_hot"),
    EncodingVariant("E2 moda + one-hot com raras agrupadas", "moda", "one_hot_raras"),
    EncodingVariant("E3 moda + ordinal", "moda", "ordinal"),
    EncodingVariant("E4 moda + frequência", "moda", "frequencia"),
    EncodingVariant("E5 moda + híbrida (one-hot; frequência em service)", "moda", "hibrida"),
)


def _categorical_pipeline(imputacao: str, codificacao: str) -> SklearnPipeline:
    return SklearnPipeline([("imputacao", _imputer(imputacao))] + _encoder_steps(codificacao))


def _imputer(imputacao: str) -> SimpleImputer:
    if imputacao == "moda":
        return SimpleImputer(strategy="most_frequent")
    if imputacao == "ausente":
        return SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)
    raise ValueError(f"imputação desconhecida: {imputacao}")


def _encoder_steps(codificacao: str) -> list[tuple[str, Any]]:
    if codificacao == "one_hot":
        return [("one_hot", OneHotEncoder(handle_unknown="ignore"))]
    if codificacao == "one_hot_raras":
        return [
            (
                "one_hot",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=RARE_MIN_FREQUENCY),
            )
        ]
    if codificacao == "ordinal":
        # Ordem arbitrária (alfabética): escalamos para não dominar a regressão logística.
        return [
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("escala", StandardScaler()),
        ]
    if codificacao == "frequencia":
        return [("frequencia", FrequencyEncoder()), ("escala", StandardScaler())]
    raise ValueError(f"codificação desconhecida: {codificacao}")


def build_encoding_preprocessor(report: PreparationReport, variant: EncodingVariant) -> ColumnTransformer:
    categorical = list(report.categorical_columns)
    numeric_pipeline = SklearnPipeline(
        steps=[
            ("imputacao", SimpleImputer(strategy="median", add_indicator=True)),
            ("padronizacao", StandardScaler(with_mean=False)),
        ]
    )

    if variant.codificacao == "hibrida" and HIGH_CARDINALITY_COLUMN in categorical:
        low_cardinality = [column for column in categorical if column != HIGH_CARDINALITY_COLUMN]
        categorical_transformers = [
            (
                "categoricos",
                _categorical_pipeline(variant.imputacao, "one_hot"),
                low_cardinality,
            ),
            (
                "service_frequencia",
                _categorical_pipeline(variant.imputacao, "frequencia"),
                [HIGH_CARDINALITY_COLUMN],
            ),
        ]
    else:
        codificacao = "one_hot" if variant.codificacao == "hibrida" else variant.codificacao
        categorical_transformers = [
            (
                "categoricos",
                _categorical_pipeline(variant.imputacao, codificacao),
                categorical,
            )
        ]

    return ColumnTransformer(
        transformers=categorical_transformers + [("numericos", numeric_pipeline, list(report.numeric_columns))],
        remainder="drop",
        sparse_threshold=1.0,
    )


def build_encoding_pipeline(report: PreparationReport, variant: EncodingVariant) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessamento", build_encoding_preprocessor(report, variant)),
            ("classificador_referencia", LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)),
        ]
    )


def encoded_dimensions(X: pd.DataFrame, report: PreparationReport) -> pd.DataFrame:
    """Colunas geradas por variante, ajustando no treino inteiro (só descritivo, não seleciona nada)."""
    rows = []
    for variant in ENCODING_VARIANTS:
        preprocessor = build_encoding_preprocessor(report, variant).fit(X)
        names = preprocessor.get_feature_names_out()
        categorical = sum(not name.startswith("numericos__") for name in names)
        rows.append(
            {
                "variante": variant.nome,
                "colunas_categoricas": int(categorical),
                "colunas_totais": int(len(names)),
            }
        )
    return pd.DataFrame(rows)


def unseen_categories_by_fold(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Quantas linhas da validação trazem categoria que não existia no treino da dobra."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    rows = []
    for fold, (train_index, valid_index) in enumerate(cv.split(X, y, make_duplicate_groups(X)), start=1):
        train, valid = X.iloc[train_index], X.iloc[valid_index]
        for column in [c for c in ("protocol_type", "service", "flag") if c in X.columns]:
            unseen = ~valid[column].isin(set(train[column].dropna())) & valid[column].notna()
            rows.append(
                {
                    "dobra": fold,
                    "atributo": column,
                    "categorias_novas": int(valid.loc[unseen, column].nunique()),
                    "linhas_afetadas": int(unseen.sum()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_encoding_strategies(
    X: pd.DataFrame,
    y: pd.Series,
    report: PreparationReport,
    *,
    n_splits: int = 5,
) -> pd.DataFrame:
    groups = make_duplicate_groups(X)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    rows: list[dict[str, Any]] = []

    for variant in ENCODING_VARIANTS:
        scores = cross_validate(
            build_encoding_pipeline(report, variant),
            X,
            y,
            groups=groups,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )
        row: dict[str, Any] = {
            "variante": variant.nome,
            "imputacao_categorica": variant.imputacao,
            "codificacao": variant.codificacao,
            "classificador_referencia": "regressao_logistica",
            "dobras": n_splits,
        }
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_media"] = float(np.mean(values))
            row[f"{metric}_desvio"] = float(np.std(values, ddof=1))
        row["tempo_ajuste_s_media"] = float(np.mean(scores["fit_time"]))
        rows.append(row)

    results = pd.DataFrame(rows).merge(encoded_dimensions(X, report), on="variante")
    return results.sort_values("f1_macro_media", ascending=False).reset_index(drop=True)
