import unittest

import pandas as pd

from src.data_validation import quality_report, validate_contract


class DataValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame(
            {
                "protocol_type": ["tcp", "udp", "tcp", "tcp"],
                "service": ["http", "dns", "http", "http"],
                "flag": ["SF", "SF", "REJ", "SF"],
                "duration": [1, 2, 3, 4],
                "class": ["normal", "anomaly", "normal", "normal"],
            }
        )
        self.test = pd.DataFrame(
            {
                "id": [10, 11],
                "protocol_type": ["tcp", "udp"],
                "service": ["http", "dns"],
                "flag": ["SF", "SF"],
                "duration": [4, 5],
            }
        )
        self.submission = pd.DataFrame(
            {"id": [10, 11], "class_prevista": ["normal", "normal"]}
        )

    def test_valid_contract_has_no_errors(self) -> None:
        self.assertEqual(validate_contract(self.train, self.test, self.submission), [])

    def test_repeated_test_id_is_rejected(self) -> None:
        self.test.loc[1, "id"] = 10
        errors = validate_contract(self.train, self.test, self.submission)
        self.assertTrue(any("ids repetidos" in error for error in errors))

    def test_unexpected_class_is_rejected(self) -> None:
        self.train.loc[0, "class"] = "probe"
        errors = validate_contract(self.train, self.test, self.submission)
        self.assertTrue(any("classes inesperadas" in error for error in errors))

    def test_quality_report_counts_duplicates(self) -> None:
        duplicated = pd.concat([self.train, self.train.iloc[[0]]], ignore_index=True)
        report = quality_report(duplicated, self.test)
        self.assertEqual(report["duplicate_train_rows"], 1)
        self.assertEqual(report["train_shape"], [5, 5])


if __name__ == "__main__":
    unittest.main()
