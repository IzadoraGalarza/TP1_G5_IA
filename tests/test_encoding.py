import unittest

import numpy as np
import pandas as pd

from src.encoding import (
    ENCODING_VARIANTS,
    FrequencyEncoder,
    encoded_dimensions,
    evaluate_encoding_strategies,
    unseen_categories_by_fold,
)
from src.preprocessing_stage2 import prepare_modeling_data


def synthetic_data(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    services = [f"svc_{i}" for i in range(12)]
    anomaly = np.arange(n) % 4 == 0
    return pd.DataFrame(
        {
            "protocol_type": rng.choice(["tcp", "udp", "icmp"], n),
            "service": np.where(anomaly, "private", rng.choice(services, n)),
            "flag": np.where(rng.random(n) < 0.1, None, rng.choice(["SF", "REJ", "S0"], n)),
            "duration": rng.integers(0, 50, n),
            "src_bytes": rng.integers(0, 5_000, n),
            "count": np.where(anomaly, 300, rng.integers(0, 20, n)),
            "class": np.where(anomaly, "anomaly", "normal"),
        }
    )


class FrequencyEncoderTests(unittest.TestCase):
    def test_uses_train_frequencies_and_zero_for_unseen(self) -> None:
        train = pd.DataFrame({"service": ["http", "http", "dns", "ssh"]})
        encoder = FrequencyEncoder().fit(train)
        encoded = encoder.transform(pd.DataFrame({"service": ["http", "dns", "novo"]}))
        np.testing.assert_allclose(encoded[:, 0], [0.5, 0.25, 0.0])


class EncodingExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.y, self.report = prepare_modeling_data(synthetic_data())

    def test_one_hot_is_wider_than_ordinal_and_frequency(self) -> None:
        dims = encoded_dimensions(self.X, self.report).set_index("variante")["colunas_categoricas"]
        one_hot = dims[ENCODING_VARIANTS[0].nome]
        self.assertGreater(one_hot, dims[ENCODING_VARIANTS[3].nome])
        self.assertEqual(dims[ENCODING_VARIANTS[3].nome], 3)
        self.assertEqual(dims[ENCODING_VARIANTS[4].nome], 3)
        self.assertGreater(dims[ENCODING_VARIANTS[1].nome], one_hot)  # nível "ausente" em flag

    def test_unseen_categories_are_counted_per_fold(self) -> None:
        unseen = unseen_categories_by_fold(self.X, self.y, n_splits=2)
        self.assertEqual(set(unseen["atributo"]), {"protocol_type", "service", "flag"})
        self.assertTrue((unseen["linhas_afetadas"] >= 0).all())

    def test_all_variants_run_under_common_protocol(self) -> None:
        results = evaluate_encoding_strategies(self.X, self.y, self.report, n_splits=2)
        self.assertEqual(len(results), len(ENCODING_VARIANTS))
        self.assertTrue(results["f1_macro_media"].between(0, 1).all())
        self.assertIn("precision_anomaly_media", results.columns)


if __name__ == "__main__":
    unittest.main()
