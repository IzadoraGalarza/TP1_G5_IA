"""Tratamento de dados redundantes e inconsistentes, ajustado somente na dobra de treino.

Duas peças, ambas encaixadas num ``imblearn.pipeline.Pipeline``:

- ``InconsistencyFixer``: transformador linha a linha (vale para treino, validação e teste);
- ``row_cleaning_sampler``: remove duplicatas/conflitos apenas no ``fit``, nunca na validação.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.data_validation import CATEGORICAL_COLUMNS
from src.experiments import POSITIVE_CLASS


RANDOM_STATE = 42

# Campos não canônicos com risco de vazamento (relatório da Etapa 1, seção 9).
SUSPECT_COLUMNS = ("tipo_alerta_soc", "score_ids_legado", "tempo_resposta_analista_min")
# Cópias determinísticas de src_bytes/dst_bytes (seção 6.3): zero inconsistências encontradas.
DERIVED_COLUMNS = ("src_bytes_kb", "dst_bytes_kb", "total_bytes")
# Faixas semânticas do NSL-KDD (janela de 2 s e janela de 100 conexões por host).
COUNT_LIMITS = {
    "count": (0, 511),
    "srv_count": (0, 511),
    "dst_host_count": (0, 255),
    "dst_host_srv_count": (0, 255),
}
NON_NEGATIVE_COLUMNS = (
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "num_compromised", "num_root", "num_file_creations",
    "num_shells", "num_access_files",
)
INVALID_STRATEGIES = ("keep", "nan", "clip")
DUPLICATE_STRATEGIES = ("keep", "drop")
CONFLICT_STRATEGIES = ("keep", "drop", "majority")


def _rate_columns(columns: pd.Index) -> list[str]:
    return [column for column in columns if column.endswith("_rate")]


class InconsistencyFixer(BaseEstimator, TransformerMixin):
    """Corrige representação e valores fora do domínio; aprende só quais colunas são constantes.

    invalid: ``keep`` não mexe; ``nan`` marca como ausente para a imputação (ajustada na dobra);
    ``clip`` traz para o limite mais próximo da faixa válida.
    """

    def __init__(
        self,
        *,
        drop_suspect: bool = True,
        normalize_categories: bool = True,
        recover_bytes: bool = True,
        drop_redundant: bool = True,
        invalid: str = "nan",
    ) -> None:
        self.drop_suspect = drop_suspect
        self.normalize_categories = normalize_categories
        self.recover_bytes = recover_bytes
        self.drop_redundant = drop_redundant
        self.invalid = invalid

    def fit(self, X: pd.DataFrame, y: Any = None) -> "InconsistencyFixer":
        if self.invalid not in INVALID_STRATEGIES:
            raise ValueError(f"invalid deve ser um de {INVALID_STRATEGIES}")
        prepared = self._prepare(X)
        self.constant_columns_ = (
            [column for column in prepared.columns if prepared[column].nunique(dropna=False) <= 1]
            if self.drop_redundant
            else []
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._prepare(X).drop(columns=self.constant_columns_, errors="ignore")

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()

        if self.normalize_categories:
            for column in CATEGORICAL_COLUMNS:
                if column in data.columns:
                    data[column] = data[column].astype("string").str.strip().str.lower().astype(object)
                    data[column] = data[column].where(data[column].notna(), np.nan)

        if self.recover_bytes:
            data = self._recover_bytes(data)

        if self.invalid != "keep":
            data = self._handle_invalid(data)

        columns_to_drop: list[str] = []
        if self.drop_suspect:
            columns_to_drop.extend(SUSPECT_COLUMNS)
        if self.drop_redundant:
            columns_to_drop.extend(DERIVED_COLUMNS)
        return data.drop(columns=columns_to_drop, errors="ignore")

    @staticmethod
    def _recover_bytes(data: pd.DataFrame) -> pd.DataFrame:
        """Preenche bytes ausentes pelas relações exatas verificadas na EDA (sem usar o rótulo)."""
        for original in ("src_bytes", "dst_bytes"):
            kb = f"{original}_kb"
            if original in data.columns and kb in data.columns:
                data[original] = data[original].fillna(data[kb] * 1024)
        if {"src_bytes", "dst_bytes", "total_bytes"}.issubset(data.columns):
            data["src_bytes"] = data["src_bytes"].fillna(data["total_bytes"] - data["dst_bytes"])
        return data

    def _handle_invalid(self, data: pd.DataFrame) -> pd.DataFrame:
        bounds: dict[str, tuple[float, float]] = dict(COUNT_LIMITS)
        bounds.update({column: (0.0, 1.0) for column in _rate_columns(data.columns)})
        bounds.update({column: (0.0, np.inf) for column in NON_NEGATIVE_COLUMNS})

        for column, (low, high) in bounds.items():
            if column not in data.columns:
                continue
            values = pd.to_numeric(data[column], errors="coerce")
            if self.invalid == "clip":
                data[column] = values.clip(lower=low, upper=high)
            else:
                data[column] = values.mask((values < low) | (values > high))
        return data


def _clean_rows(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    duplicates: str,
    conflicts: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Resolve conflitos de rótulo e remove duplicatas exatas, só no conjunto recebido."""
    y = pd.Series(np.asarray(y), index=X.index)
    key = pd.util.hash_pandas_object(X, index=False)

    if conflicts != "keep":
        labels_per_key = y.groupby(key).transform("nunique")
        conflicting = labels_per_key > 1
        if conflicts == "drop":
            X, y, key = X[~conflicting], y[~conflicting], key[~conflicting]
        else:
            y = y.copy()
            y[conflicting] = y[conflicting].groupby(key[conflicting]).transform(_majority_label)

    if duplicates == "drop":
        keep = ~pd.DataFrame({"key": key, "label": y}).duplicated()
        X, y = X[keep], y[keep]

    return X, y


def _majority_label(labels: pd.Series) -> str:
    counts = labels.value_counts()
    winners = counts[counts == counts.max()].index
    # Empate: favorece a classe rara, pois deixar de detectar um ataque custa mais.
    return POSITIVE_CLASS if POSITIVE_CLASS in winners else winners[0]


def row_cleaning_sampler(*, duplicates: str = "drop", conflicts: str = "keep") -> FunctionSampler:
    if duplicates not in DUPLICATE_STRATEGIES:
        raise ValueError(f"duplicates deve ser um de {DUPLICATE_STRATEGIES}")
    if conflicts not in CONFLICT_STRATEGIES:
        raise ValueError(f"conflicts deve ser um de {CONFLICT_STRATEGIES}")
    return FunctionSampler(
        func=_clean_rows,
        validate=False,
        kw_args={"duplicates": duplicates, "conflicts": conflicts},
    )


@dataclass(frozen=True)
class CleaningConfig:
    drop_suspect: bool = True
    normalize_categories: bool = True
    recover_bytes: bool = True
    drop_redundant: bool = True
    invalid: str = "nan"
    duplicates: str = "drop"
    conflicts: str = "keep"

    def with_changes(self, **changes: Any) -> "CleaningConfig":
        return replace(self, **changes)


# Configuração de partida para as Partes 2 a 5; confirmar/atualizar com o resultado do notebook 02.
LIMPEZA_PADRAO = CleaningConfig()


def cleaning_steps(config: CleaningConfig = LIMPEZA_PADRAO) -> list[tuple[str, Any]]:
    """Passos de limpeza para compor o pipeline das outras partes."""
    fixer = InconsistencyFixer(
        drop_suspect=config.drop_suspect,
        normalize_categories=config.normalize_categories,
        recover_bytes=config.recover_bytes,
        drop_redundant=config.drop_redundant,
        invalid=config.invalid,
    )
    return [
        ("limpeza", fixer),
        ("linhas", row_cleaning_sampler(duplicates=config.duplicates, conflicts=config.conflicts)),
    ]


def default_encoding() -> ColumnTransformer:
    """Imputação + one-hot iguais ao baseline da Parte 1, selecionando colunas pelo tipo."""
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, make_column_selector(dtype_include=object)),
            ("numeric", numeric, make_column_selector(dtype_include="number")),
        ],
        remainder="drop",
    )


def build_cleaning_pipeline(
    config: CleaningConfig = LIMPEZA_PADRAO,
    classifier: Any | None = None,
) -> Pipeline:
    classifier = classifier if classifier is not None else DecisionTreeClassifier(random_state=RANDOM_STATE)
    return Pipeline(
        steps=cleaning_steps(config) + [("codificacao", default_encoding()), ("classificador", classifier)]
    )


# Variantes comparadas no notebook 02: cada uma muda uma decisão em relação à anterior.
CLEANING_VARIANTS: dict[str, CleaningConfig] = {
    "C0 sem tratamento": CleaningConfig(
        normalize_categories=False, recover_bytes=False, drop_redundant=False,
        invalid="keep", duplicates="keep",
    ),
    "C1 + categorias normalizadas e redundantes removidos": CleaningConfig(
        recover_bytes=False, invalid="keep", duplicates="keep",
    ),
    "C2 + bytes recuperados pelas relações exatas": CleaningConfig(invalid="keep", duplicates="keep"),
    "C3 + inválidos -> NaN + imputação": CleaningConfig(duplicates="keep"),
    "C3b + inválidos -> clip na faixa": CleaningConfig(invalid="clip", duplicates="keep"),
    "C4 + duplicatas exatas removidas": CleaningConfig(),
    "C5 + conflitos removidos": CleaningConfig(conflicts="drop"),
    "C5b + conflitos por voto majoritário": CleaningConfig(conflicts="majority"),
}
# Sensibilidade apenas: mede o quanto as colunas suspeitas inflam o F1. Não é candidata.
SUSPECT_SENSITIVITY = ("S  colunas suspeitas mantidas (NÃO elegível)", CleaningConfig(drop_suspect=False))
