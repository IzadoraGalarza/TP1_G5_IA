"""Validação do contrato e diagnóstico inicial dos CSVs da competição."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_COLUMN = "class"
ID_COLUMN = "id"
PREDICTION_COLUMN = "class_prevista"
ALLOWED_CLASSES = frozenset({"normal", "anomaly"})
CATEGORICAL_COLUMNS = ("protocol_type", "service", "flag")


@dataclass(frozen=True)
class DatasetPaths:
    train: Path
    test: Path
    submission_model: Path

    @classmethod
    def from_directory(cls, data_dir: Path) -> "DatasetPaths":
        return cls(
            train=data_dir / "treino.csv",
            test=data_dir / "teste_sem_rotulo.csv",
            submission_model=data_dir / "modelo_de_submissao.csv",
        )


def load_competition_data(paths: DatasetPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_files = [str(path) for path in asdict(paths).values() if not Path(path).is_file()]
    if missing_files:
        formatted = "\n- ".join(missing_files)
        raise FileNotFoundError(f"Arquivos obrigatórios ausentes:\n- {formatted}")

    return (
        pd.read_csv(paths.train),
        pd.read_csv(paths.test),
        pd.read_csv(paths.submission_model),
    )


def validate_contract(
    train: pd.DataFrame,
    test: pd.DataFrame,
    submission_model: pd.DataFrame,
) -> list[str]:
    """Retorna erros de contrato; lista vazia significa contrato válido."""
    errors: list[str] = []

    if TARGET_COLUMN not in train.columns:
        errors.append(f"treino sem a coluna alvo '{TARGET_COLUMN}'")
    if ID_COLUMN not in test.columns:
        errors.append(f"teste sem a coluna '{ID_COLUMN}'")
    if TARGET_COLUMN in test.columns:
        errors.append("teste sem rótulo contém a coluna alvo")

    expected_submission_columns = [ID_COLUMN, PREDICTION_COLUMN]
    if list(submission_model.columns) != expected_submission_columns:
        errors.append(
            "modelo de submissão deve ter exatamente as colunas "
            f"{expected_submission_columns}, nessa ordem"
        )

    if TARGET_COLUMN in train.columns:
        target = train[TARGET_COLUMN]
        observed_classes = set(target.dropna().astype(str).str.lower().unique())
        unexpected = observed_classes - ALLOWED_CLASSES
        if unexpected:
            errors.append(f"classes inesperadas no treino: {sorted(unexpected)}")
        if target.isna().any():
            errors.append("há rótulos ausentes no treino")

    if ID_COLUMN in test.columns:
        if test[ID_COLUMN].isna().any():
            errors.append("há ids ausentes no teste")
        if test[ID_COLUMN].duplicated().any():
            errors.append("há ids repetidos no teste")

    if ID_COLUMN in submission_model.columns and ID_COLUMN in test.columns:
        if len(submission_model) != len(test):
            errors.append("modelo de submissão e teste têm números de linhas diferentes")
        elif not submission_model[ID_COLUMN].equals(test[ID_COLUMN]):
            errors.append("ids do modelo de submissão não coincidem com o teste, na mesma ordem")

    train_features = set(train.columns) - {TARGET_COLUMN, ID_COLUMN}
    test_features = set(test.columns) - {ID_COLUMN}
    if train_features != test_features:
        only_train = sorted(train_features - test_features)
        only_test = sorted(test_features - train_features)
        errors.append(
            f"atributos divergentes; apenas no treino={only_train}, apenas no teste={only_test}"
        )

    missing_categorical = sorted(set(CATEGORICAL_COLUMNS) - train_features)
    if missing_categorical:
        errors.append(f"atributos categóricos declarados ausentes: {missing_categorical}")

    return errors


def quality_report(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Produz um resumo serializável para orientar a EDA, sem ajustar transformadores."""
    feature_columns = [column for column in train.columns if column not in {TARGET_COLUMN, ID_COLUMN}]
    constant_columns = [column for column in feature_columns if train[column].nunique(dropna=False) <= 1]
    categorical_cardinality = {
        column: int(train[column].nunique(dropna=True))
        for column in CATEGORICAL_COLUMNS
        if column in train.columns
    }

    target_counts: dict[str, int] = {}
    if TARGET_COLUMN in train.columns:
        target_counts = {
            str(label): int(count)
            for label, count in train[TARGET_COLUMN].value_counts(dropna=False).items()
        }

    return {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "target_counts": target_counts,
        "missing_train": int(train.isna().sum().sum()),
        "missing_test": int(test.isna().sum().sum()),
        "duplicate_train_rows": int(train.duplicated().sum()),
        "duplicate_train_features": int(train.duplicated(subset=feature_columns).sum()),
        "constant_columns": constant_columns,
        "categorical_cardinality": categorical_cardinality,
        "numeric_feature_count": int(train[feature_columns].select_dtypes(include="number").shape[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    paths = DatasetPaths.from_directory(args.data_dir)
    try:
        train, test, submission_model = load_competition_data(paths)
    except FileNotFoundError as error:
        parser.error(str(error))

    errors = validate_contract(train, test, submission_model)
    print(json.dumps(quality_report(train, test), indent=2, ensure_ascii=False))
    if errors:
        print("\nErros de contrato:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nContrato dos dados válido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
