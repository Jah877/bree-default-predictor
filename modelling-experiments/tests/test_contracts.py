"""Deterministic checks for the dataset, baseline, and evaluation contract."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data-analysis"))
sys.path.insert(0, str(ROOT / "modelling-experiments"))

from analysis import validate_and_load
from baseline import compare_to_stored, score_components
from splits import make_splits
from evaluation import score_binary, select_threshold_at_intervention_rate
from run_experiments import get_X, labels
from specs import make_pipeline
from experiments.e2_engineered_logistic import SPEC as E2


class SubmissionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "config.yaml").open(encoding="utf-8") as file:
            cls.settings = yaml.safe_load(file)
        cls.frame = validate_and_load(cls.settings)

    def test_dataset_original_shape_and_labels(self):
        self.assertEqual(self.frame.shape, (2000, 15))
        self.assertEqual(self.frame.actual_outcome.value_counts().to_dict(),
                         {"repaid": 1291, "defaulted": 545, "ongoing": 164})
        self.assertEqual(self.frame.applicant_id.nunique(), 2000)

    def test_reconstructed_baseline_matches_all_actions(self):
        check = compare_to_stored(self.frame, score_components(self.frame))
        self.assertEqual(check["policy_decision_matches"], 2000)
        self.assertEqual(check["exact_score_matches"], 1979)

    def test_only_application_time_features_are_sent_to_models(self):
        raw = get_X(self.frame)
        self.assertEqual(set(raw.columns), {
            "applicant_id", "stated_monthly_income", "documented_monthly_income",
            "loan_amount", "bank_ending_balance", "bank_has_overdrafts",
            "bank_has_consistent_deposits", "monthly_withdrawals",
            "monthly_deposits", "num_documents_submitted", "employment_status"
        })
        self.assertTrue(raw.applicant_id.equals(self.frame.applicant_id))

    def test_metrics_confusion_orientation(self):
        y = np.array([0, 0, 1, 1])
        flags = np.array([0, 1, 0, 1])
        result = score_binary(y, np.array([0.1, 0.8, 0.2, 0.9]),
                              flags, "example", "unit", 0.5, True)
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]),
                         (1, 1, 1, 1))
        self.assertAlmostEqual(result["default_recall"], 0.5)
        self.assertAlmostEqual(result["false_positive_rate"], 0.5)

    def test_tree_tied_scores_cannot_always_match_intervention_rate(self):
        p = np.array([0.1, 0.1, 0.7, 0.7])
        threshold = select_threshold_at_intervention_rate(p, 0.25)
        self.assertIn(int((p >= threshold).sum()), (0, 2))

    def test_pipeline_training_outputs_probabilities_without_leakage(self):
        parts = make_splits(self.frame, self.settings)
        model = make_pipeline(E2)
        model.fit(get_X(parts["train"]), labels(parts["train"], self.settings))
        test_prob = model.predict_proba(get_X(parts["validation"]))[:, 1]
        self.assertEqual(test_prob.shape, (len(parts["validation"]),))
        self.assertTrue(np.isfinite(test_prob).all())
        self.assertTrue(np.all((test_prob >= 0) & (test_prob <= 1)))
        # No records in the validation set can be learned from through the
        # imputer or scaler: those stats are estimated only during .fit(train).
        imputer = model.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["impute"]
        self.assertEqual(len(imputer.statistics_), 13)


if __name__ == "__main__":
    unittest.main()
