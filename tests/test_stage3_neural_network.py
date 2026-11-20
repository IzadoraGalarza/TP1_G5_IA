import unittest
import warnings

import numpy as np
import pandas as pd

from src.stage3_neural_network import (
    LEAKAGE_SUSPECTS,
    MLPConfig,
    REDUNDANT_COLUMNS,
    clean_stage2_data,
    evaluate_configuration,
    split_features_target,
)


class Stage3NeuralNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for index in range(60):
            anomaly = index % 4 == 0
            rows.append(
                {
                    "protocol_type": " TCP " if index % 2 else "udp",
                    "service": "http" if index % 3 else "dns",
                    "flag": "SF" if index % 2 else "REJ",
                    "duration": -1 if index == 0 else index,
                    "src_bytes": np.nan if index % 7 == 0 else index * 10,
                    "dst_bytes": index,
                    "count": 512_000 if index == 1 else index,
                    "rerror_rate": 10 if index == 2 else index / 100,
                    "src_bytes_kb": index / 1024,
                    "dst_bytes_kb": index / 1024,
                    "total_bytes": index * 11,
                    "same_srv_pct": 50,
                    "protocolo_cod": 1,
                    "num_outbound_cmds": 0,
                    "is_host_login": 0,
                    "tipo_alerta_soc": "ALERTA" if anomaly else "OK",
                    "score_ids_legado": 0.9 if anomaly else 0.1,
                    "tempo_resposta_analista_min": 30 if anomaly else np.nan,
                    "class": "anomaly" if anomaly else "normal",
                }
            )
        self.data = pd.DataFrame(rows)

    def test_cleaning_matches_stage2_decisions(self) -> None:
        cleaned = clean_stage2_data(self.data)
        for column in (*LEAKAGE_SUSPECTS, *REDUNDANT_COLUMNS):
            self.assertNotIn(column, cleaned.columns)
        self.assertEqual(cleaned["protocol_type"].iloc[1], "tcp")
        self.assertTrue(pd.isna(cleaned.loc[0, "duration"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "count"]))

    def test_binary_target_and_small_mlp_run(self) -> None:
        X, y = split_features_target(self.data)
        self.assertEqual(set(y.unique()), {0, 1})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summary, folds = evaluate_configuration(
                X,
                y,
                MLPConfig((8,), 0.001, True),
                n_splits=2,
                max_iter=10,
            )
        self.assertEqual(len(folds), 2)
        self.assertGreaterEqual(summary["f1_macro"], 0)
        self.assertLessEqual(summary["f1_macro"], 1)
        for key in (
            "precision_normal",
            "recall_normal",
            "precision_anomaly",
            "recall_anomaly",
        ):
            self.assertIn(key, summary)


if __name__ == "__main__":
    unittest.main()
