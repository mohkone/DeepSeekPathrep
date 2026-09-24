"""Small synthetic safeguards for the deduplicated survival workflow."""

import unittest

import numpy as np
import pandas as pd

from run_cox_deduplicated import bootstrap, prepare_cohorts, text_hash


def rows(ids, texts):
    return pd.DataFrame({
        "bcr_patient_barcode": ids,
        "report_text": texts,
        "DSS.time": range(1, len(ids) + 1),
        "dss_event": [1] * len(ids),
    })


class DeduplicationTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(text_hash(" A \n B "), text_hash("a b"))
        self.assertNotEqual(text_hash("a b"), text_hash("a c"))

    def test_patient_and_report_dedup_and_cohort_lock(self):
        split = {
            "train": rows(["A", "B", "C"], ["One", "Two", "Three"]),
            "val": rows(["A", "D", "E"], ["One", "Four", "Two"]),
            "test": rows(["B", "F", "G"], ["Two", " Four ", "Unique"]),
        }
        train, test, counts = prepare_cohorts(split)
        self.assertEqual(counts["patient_duplicates_removed"], 1)
        self.assertEqual(counts["same_text_different_patient_removed"], 1)
        self.assertEqual(train["bcr_patient_barcode"].tolist(), ["A", "B", "C", "D"])
        self.assertEqual(test["id_disjoint"].tolist(), [False, True, True])
        self.assertEqual(test["id_text_disjoint"].tolist(), [False, False, True])

    def test_bootstrap_is_paired_and_deterministic(self):
        frame = rows(["A", "B", "C", "D"], ["a", "b", "c", "d"])
        risk = np.array([4., 3., 2., 1.])
        prediction = {"clinical_only": risk, "clinical_plus_text": risk.copy()}
        mask = np.ones(4, dtype=bool)
        result = bootstrap(frame, prediction, mask, 100, 20260924)
        self.assertEqual(result["resamples_requested"], 100)
        self.assertEqual(
            result["percentile_95_ci"]["clinical_plus_text_minus_clinical_only"],
            [0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
