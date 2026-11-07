import unittest

import pandas as pd

from src.baseline import evaluate_baseline, split_features_target


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame(
            {
                "id": range(12),
                "protocol_type": ["tcp", "udp"] * 6,
                "service": ["http", "dns", "ssh"] * 4,
                "flag": ["SF", "REJ"] * 6,
                "duration": list(range(12)),
                "class": ["normal", "anomaly"] * 6,
            }
        )

    def test_split_removes_id_and_target(self) -> None:
        X, y = split_features_target(self.train)
        self.assertNotIn("id", X.columns)
        self.assertNotIn("class", X.columns)
        self.assertEqual(len(y), len(self.train))

    def test_evaluation_returns_one_row_per_fold(self) -> None:
        results, report = evaluate_baseline(self.train, n_splits=3)
        self.assertEqual(len(results), 3)
        self.assertIn("macro avg", report)
        self.assertTrue(results["f1_macro"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
