"""Experimentos da Rede Neural da Etapa 3, alinhados ao Colab da equipe."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
TARGET = "class"
ID_COLUMN = "id"
REDUNDANT_COLUMNS = (
    "src_bytes_kb",
    "dst_bytes_kb",
    "total_bytes",
    "same_srv_pct",
    "protocolo_cod",
    "num_outbound_cmds",
    "is_host_login",
)
LEAKAGE_SUSPECTS = (
    "tipo_alerta_soc",
    "score_ids_legado",
    "tempo_resposta_analista_min",
)
COUNT_RANGES = {
    "count": (0, 511),
    "srv_count": (0, 511),
    "dst_host_count": (0, 255),
    "dst_host_srv_count": (0, 255),
}
NONNEGATIVE_COLUMNS = ("duration", "src_bytes", "dst_bytes")


@dataclass(frozen=True)
class MLPConfig:
    hidden_layer_sizes: tuple[int, ...]
    alpha: float
    early_stopping: bool

    @property
    def architecture(self) -> str:
        return "x".join(map(str, self.hidden_layer_sizes))


def binary_target(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    if values.isna().any():
        raise ValueError("a classe possui valores ausentes")
    unknown = sorted(set(values.unique()) - {"normal", "anomaly"})
    if unknown:
        raise ValueError(f"classes inesperadas: {unknown}")
    return (values == "anomaly").astype(int)


def normalize_categories(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    for column in ("protocol_type", "service", "flag"):
        if column in normalized.columns:
            normalized[column] = normalized[column].astype(object).map(
                lambda value: value.strip().lower() if isinstance(value, str) else value
            )
    return normalized


def mask_out_of_range(data: pd.DataFrame) -> pd.DataFrame:
    masked = data.copy()
    for column, (lower, upper) in COUNT_RANGES.items():
        if column in masked.columns:
            values = pd.to_numeric(masked[column], errors="coerce")
            masked.loc[(values < lower) | (values > upper), column] = np.nan
    for column in [name for name in masked.columns if name.lower().endswith("_rate")]:
        values = pd.to_numeric(masked[column], errors="coerce")
        masked.loc[(values < 0) | (values > 1), column] = np.nan
    for column in NONNEGATIVE_COLUMNS:
        if column in masked.columns:
            values = pd.to_numeric(masked[column], errors="coerce")
            masked.loc[values < 0, column] = np.nan
    return masked


def clean_stage2_data(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduz a limpeza F_E_mais_suspeitas escolhida no Colab da Etapa 2."""
    if TARGET not in data.columns:
        raise ValueError(f"coluna alvo ausente: {TARGET}")

    cleaned = data.drop(
        columns=[
            column
            for column in (*REDUNDANT_COLUMNS, *LEAKAGE_SUSPECTS, ID_COLUMN)
            if column in data.columns
        ]
    ).copy()
    cleaned = normalize_categories(cleaned)
    cleaned = mask_out_of_range(cleaned)
    cleaned = cleaned.dropna(subset=[TARGET]).drop_duplicates().copy()

    features = [column for column in cleaned.columns if column != TARGET]
    if features:
        conflicting = (
            cleaned.groupby(features, dropna=False)[TARGET].transform("nunique") > 1
        )
        cleaned = cleaned.loc[~conflicting].copy()
    return cleaned.reset_index(drop=True)


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cleaned = clean_stage2_data(data)
    return cleaned.drop(columns=[TARGET]), binary_target(cleaned[TARGET])


def make_onehot_rare() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=0.001,
            sparse_output=False,
        )
    except TypeError:  # scikit-learn anterior a 1.2
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=0.001,
            sparse=False,
        )


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric = X.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "categoricos",
                Pipeline(
                    [
                        ("imputacao", SimpleImputer(strategy="most_frequent")),
                        ("onehot_rare", make_onehot_rare()),
                    ]
                ),
                categorical,
            ),
            (
                "numericos",
                Pipeline(
                    [
                        ("imputacao", SimpleImputer(strategy="median")),
                        ("normalizacao", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
    )


def build_pipeline(
    X: pd.DataFrame,
    config: MLPConfig,
    *,
    max_iter: int = 120,
) -> Pipeline:
    model = MLPClassifier(
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=config.alpha,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=max_iter,
        early_stopping=config.early_stopping,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
        random_state=SEED,
    )
    return Pipeline([("preprocessamento", make_preprocessor(X)), ("rede_neural", model)])


def signature_groups(X: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(X.astype(str), index=False).to_numpy()


SCORING = {
    "f1_macro": "f1_macro",
    "precision_normal": make_scorer(
        precision_score, pos_label=0, zero_division=0
    ),
    "recall_normal": make_scorer(recall_score, pos_label=0, zero_division=0),
    "precision_anomaly": make_scorer(
        precision_score, pos_label=1, zero_division=0
    ),
    "recall_anomaly": make_scorer(recall_score, pos_label=1, zero_division=0),
}


def default_grid() -> list[MLPConfig]:
    return [
        MLPConfig(hidden, alpha, early)
        for hidden, alpha, early in product(
            [(64,), (64, 32)],
            [0.0001, 0.001, 0.01],
            [True, False],
        )
    ]


def evaluate_configuration(
    X: pd.DataFrame,
    y: pd.Series,
    config: MLPConfig,
    *,
    n_splits: int = 3,
    max_iter: int = 120,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    started = time.time()
    scores = cross_validate(
        build_pipeline(X, config, max_iter=max_iter),
        X,
        y,
        groups=signature_groups(X),
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        return_estimator=True,
        error_score="raise",
    )
    elapsed = time.time() - started
    row: dict[str, object] = {
        "arquitetura": config.architecture,
        "hidden_layer_sizes": repr(config.hidden_layer_sizes),
        "alpha": config.alpha,
        "early_stopping": config.early_stopping,
        "normalizacao": "StandardScaler",
        "codificacao": "onehot_rare",
        "limpeza": "F_E_mais_suspeitas",
        "n_linhas": len(X),
        "n_dobras": n_splits,
        "tempo_s": elapsed,
    }
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        row[metric] = float(np.mean(values))
        row[f"{metric}_std"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
    iterations = [
        int(estimator.named_steps["rede_neural"].n_iter_)
        for estimator in scores["estimator"]
    ]
    row["iteracoes_media"] = float(np.mean(iterations))

    folds: list[dict[str, object]] = []
    for index in range(n_splits):
        fold = {
            "arquitetura": config.architecture,
            "alpha": config.alpha,
            "early_stopping": config.early_stopping,
            "dobra": index + 1,
            "iteracoes": iterations[index],
            "tempo_ajuste_s": float(scores["fit_time"][index]),
        }
        for metric in SCORING:
            fold[metric] = float(scores[f"test_{metric}"][index])
        folds.append(fold)
    return row, folds


def run_grid(
    X: pd.DataFrame,
    y: pd.Series,
    configs: Iterable[MLPConfig] | None = None,
    *,
    n_splits: int = 3,
    max_iter: int = 120,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = list(configs or default_grid())
    summaries: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for index, config in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] arquitetura={config.architecture}, "
            f"alpha={config.alpha:g}, early_stopping={config.early_stopping}",
            flush=True,
        )
        summary, folds = evaluate_configuration(
            X,
            y,
            config,
            n_splits=n_splits,
            max_iter=max_iter,
        )
        summaries.append(summary)
        fold_rows.extend(folds)
        if output_dir is not None:
            pd.DataFrame(summaries).to_csv(
                output_dir / "resultados_rede_neural.csv", index=False
            )
            pd.DataFrame(fold_rows).to_csv(
                output_dir / "resultados_rede_neural_por_dobra.csv", index=False
            )
        print(
            f"    F1-macro={summary['f1_macro']:.4f} ± "
            f"{summary['f1_macro_std']:.4f}",
            flush=True,
        )

    results = pd.DataFrame(summaries).sort_values(
        ["f1_macro", "recall_anomaly"], ascending=False
    ).reset_index(drop=True)
    folds = pd.DataFrame(fold_rows)
    if output_dir is not None:
        results.to_csv(output_dir / "resultados_rede_neural.csv", index=False)
        folds.to_csv(output_dir / "resultados_rede_neural_por_dobra.csv", index=False)
        best = results.iloc[0].to_dict()
        (output_dir / "melhor_configuracao_rede_neural.json").write_text(
            json.dumps(best, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        create_plots(results, output_dir)
    return results, folds


def create_plots(results: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data = results.copy()
    plot_data["configuracao"] = (
        plot_data["arquitetura"]
        + " | α="
        + plot_data["alpha"].map(lambda value: f"{value:g}")
        + " | ES="
        + plot_data["early_stopping"].map({True: "sim", False: "não"})
    )
    plot_data = plot_data.sort_values("f1_macro")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        plot_data["configuracao"],
        plot_data["f1_macro"],
        xerr=plot_data["f1_macro_std"],
        color="#d95f3d",
        ecolor="#333333",
        capsize=3,
    )
    lower = max(0.0, float(plot_data["f1_macro"].min()) - 0.01)
    upper = min(1.0, float(plot_data["f1_macro"].max()) + 0.008)
    ax.set_xlim(lower, upper)
    ax.set_xlabel("F1-macro médio (3 dobras)")
    ax.set_ylabel("Configuração da MLP")
    ax.set_title("Rede Neural: comparação de arquitetura, alpha e early stopping")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "rede_neural_f1_macro.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for early, marker, label in [(True, "o", "early stopping"), (False, "s", "sem early stopping")]:
        subset = results[results["early_stopping"] == early]
        ax.scatter(
            subset["recall_anomaly"],
            subset["precision_anomaly"],
            marker=marker,
            s=70,
            label=label,
            alpha=0.8,
        )
    ax.set_xlabel("Recall - anomaly")
    ax.set_ylabel("Precisão - anomaly")
    ax.set_title("Precisão versus recall da classe anomaly")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rede_neural_precisao_recall_anomaly.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/treino.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/etapa_3"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="executa uma única configuração para diagnóstico",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = pd.read_csv(args.train)
    X, y = split_features_target(raw)
    print("Base limpa:", X.shape)
    print("Classes:", y.value_counts().sort_index().to_dict())
    configs = [MLPConfig((16,), 0.001, True)] if args.quick else default_grid()
    results, _ = run_grid(
        X,
        y,
        configs,
        n_splits=2 if args.quick else args.folds,
        max_iter=20 if args.quick else args.max_iter,
        output_dir=args.output,
    )
    print("\nMelhores configurações:")
    print(
        results[
            [
                "arquitetura",
                "alpha",
                "early_stopping",
                "f1_macro",
                "precision_normal",
                "recall_normal",
                "precision_anomaly",
                "recall_anomaly",
            ]
        ].head(5).to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
