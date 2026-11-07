import unittest

import numpy as np
import pandas as pd

from src.cleaning import (
    CLEANING_VARIANTS,
    CleaningConfig,
    InconsistencyFixer,
    _clean_rows,
    build_cleaning_pipeline,
)
from src.experiments import make_cv, results_table, run_experiment


class InconsistencyFixerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = pd.DataFrame(
            {
                "protocol_type": ["TCP", " tcp", "udp"],
                "flag": ["SF", "sf", "REJ"],
                "duration": [-1, 3, 5],
                "count": [10, 512000, 20],
                "rerror_rate": [0.5, 10.0, 0.0],
                "src_bytes": [np.nan, np.nan, 100.0],
                "src_bytes_kb": [2.0, np.nan, 100 / 1024],
                "dst_bytes": [0.0, 50.0, 0.0],
                "total_bytes": [2048.0, 80.0, 100.0],
                "num_outbound_cmds": [0, 0, 0],
                "tipo_alerta_soc": ["OK", "ALERTA", "OK"],
            }
        )

    def test_default_fixes_representation_and_domain(self) -> None:
        result = InconsistencyFixer().fit_transform(self.X)

        self.assertEqual(result["protocol_type"].tolist(), ["tcp", "tcp", "udp"])
        self.assertEqual(result["flag"].tolist(), ["sf", "sf", "rej"])
        self.assertEqual(result["src_bytes"].tolist(), [2048.0, 30.0, 100.0])
        self.assertTrue(np.isnan(result.loc[0, "duration"]))
        self.assertTrue(np.isnan(result.loc[1, "count"]))
        self.assertTrue(np.isnan(result.loc[1, "rerror_rate"]))
        for dropped in ("num_outbound_cmds", "tipo_alerta_soc", "src_bytes_kb", "total_bytes"):
            self.assertNotIn(dropped, result.columns)

    def test_clip_brings_values_to_valid_range(self) -> None:
        result = InconsistencyFixer(invalid="clip").fit_transform(self.X)
        self.assertEqual(result.loc[0, "duration"], 0)
        self.assertEqual(result.loc[1, "count"], 511)
        self.assertEqual(result.loc[1, "rerror_rate"], 1.0)

    def test_constant_columns_are_learned_from_fit_only(self) -> None:
        fixer = InconsistencyFixer().fit(self.X)
        other = self.X.assign(num_outbound_cmds=[1, 2, 3])
        self.assertNotIn("num_outbound_cmds", fixer.transform(other).columns)


class RowCleaningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = pd.DataFrame({"a": [1, 1, 2, 2, 2, 3], "b": ["x", "x", "y", "y", "y", "z"]})
        self.y = pd.Series(["normal", "normal", "normal", "anomaly", "anomaly", "normal"])

    def test_drop_exact_duplicates_keeps_conflicts(self) -> None:
        X, y = _clean_rows(self.X, self.y, duplicates="drop", conflicts="keep")
        self.assertEqual(len(X), 4)
        self.assertEqual(sorted(y.tolist()), ["anomaly", "normal", "normal", "normal"])

    def test_drop_conflicting_groups(self) -> None:
        X, _ = _clean_rows(self.X, self.y, duplicates="keep", conflicts="drop")
        self.assertNotIn(2, X["a"].tolist())
        self.assertEqual(len(X), 3)

    def test_majority_relabels_then_deduplicates(self) -> None:
        X, y = _clean_rows(self.X, self.y, duplicates="drop", conflicts="majority")
        self.assertEqual(len(X), 3)
        self.assertEqual(y[X["a"] == 2].tolist(), ["anomaly"])


class CleaningPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        n = 60
        self.y = pd.Series(np.where(np.arange(n) % 4 == 0, "anomaly", "normal"))
        self.X = pd.DataFrame(
            {
                "protocol_type": rng.choice(["tcp", "TCP", "udp"], n),
                "service": rng.choice(["http", "dns", "ssh"], n),
                "flag": rng.choice(["SF", "REJ"], n),
                "duration": rng.integers(-1, 10, n),
                "count": np.where(self.y == "anomaly", 400, 5),
                "rerror_rate": rng.random(n),
                "num_outbound_cmds": 0,
            }
        )

    def test_dedup_does_not_touch_validation_predictions(self) -> None:
        pipeline = build_cleaning_pipeline(CleaningConfig(duplicates="drop"))
        pipeline.fit(self.X, self.y)
        self.assertEqual(len(pipeline.predict(self.X)), len(self.X))

    def test_every_variant_runs_under_common_protocol(self) -> None:
        cv = make_cv(self.y, n_splits=3)
        rows = [
            run_experiment(build_cleaning_pipeline(config), self.X, self.y, etapa="limpeza", variante=name, cv=cv)
            for name, config in CLEANING_VARIANTS.items()
        ]
        table = results_table(rows)
        self.assertEqual(len(table), len(CLEANING_VARIANTS))
        self.assertTrue(table["f1_macro_media"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
