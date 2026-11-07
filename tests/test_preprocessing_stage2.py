import unittest

import pandas as pd

from src.preprocessing_stage2 import (
    build_preprocessor,
    evaluate_cleaning_strategies,
    evaluate_preprocessing_strategies,
    make_duplicate_groups,
    prepare_modeling_data,
)


class Stage2PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for index in range(30):
            anomaly = index % 3 == 0
            rows.append(
                {
                    "protocol_type": " TCP " if index % 2 else "udp",
                    "service": "http" if index % 3 else "dns",
                    "flag": "SF" if index % 2 else "REJ",
                    "duration": -1 if index == 0 else index,
                    "src_bytes": None if index % 7 == 0 else index * 10,
                    "count": 512_000 if index == 1 else index,
                    "src_bytes_kb": index / 1024,
                    "total_bytes": index * 10,
                    "num_outbound_cmds": 0,
                    "tipo_alerta_soc": "ALERTA" if anomaly else "OK",
                    "class": "anomalia" if anomaly else "normal",
                }
            )
        self.data = pd.DataFrame(rows)

    def test_preparation_applies_stage1_decisions(self) -> None:
        X, y, report = prepare_modeling_data(self.data)
        self.assertNotIn("tipo_alerta_soc", X.columns)
        self.assertNotIn("num_outbound_cmds", X.columns)
        self.assertNotIn("src_bytes_kb", X.columns)
        self.assertEqual(set(y.unique()), {"normal", "anomaly"})
        self.assertGreaterEqual(report.invalid_values_replaced_by_nan, 2)
        self.assertEqual(X["protocol_type"].iloc[1], "tcp")

    def test_missing_categories_reach_the_imputer(self) -> None:
        self.data.loc[[2, 4], "service"] = None
        X, _, report = prepare_modeling_data(self.data)
        self.assertEqual(int(X["service"].isna().sum()), 2)
        build_preprocessor(report).fit(X)

    def test_equal_predictors_receive_equal_group(self) -> None:
        X, _, _ = prepare_modeling_data(self.data)
        duplicated = pd.concat([X, X.iloc[[0]]], ignore_index=True)
        groups = make_duplicate_groups(duplicated)
        self.assertEqual(groups.iloc[0], groups.iloc[-1])

    def test_three_balance_strategies_are_compared(self) -> None:
        X, y, report = prepare_modeling_data(self.data)
        results = evaluate_preprocessing_strategies(
            X,
            y,
            report,
            n_splits=2,
            n_components=2,
        )
        self.assertEqual(
            set(results["estrategia"]),
            {"sem_balanceamento", "random_under_sampler", "random_over_sampler"},
        )
        self.assertTrue(results["f1_macro_media"].between(0, 1).all())

    def test_cleaning_variants_are_compared(self) -> None:
        results = evaluate_cleaning_strategies(self.data, n_splits=2)
        self.assertGreaterEqual(len(results), 5)
        self.assertTrue(results["f1_macro_media"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
