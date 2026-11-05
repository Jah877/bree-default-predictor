"""Run with: python -m unittest discover -s modelling-experiments/tests -v"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modelling-experiments"))
sys.path.insert(0, str(ROOT / "data-analysis"))
from features import FeatureBuilder
from evaluation import select_threshold_at_intervention_rate
from specs import without_employment
from experiments.e2_engineered_logistic import SPEC as E2
from splits import make_splits


class MethodologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "config.yaml").open() as h:
            cls.cfg = yaml.safe_load(h)
        cls.frame = pd.read_csv(ROOT / cls.cfg["paths"]["dataset"])

    def test_split_disjoint_and_reproducible(self):
        a = make_splits(self.frame, self.cfg)
        b = make_splits(self.frame, self.cfg)
        ids = [set(a[k].applicant_id) for k in ("train", "validation", "test")]
        self.assertFalse(ids[0] & ids[1] or ids[1] & ids[2] or ids[0] & ids[2])
        self.assertEqual(sum(map(len, ids)), self.frame.actual_outcome.ne("ongoing").sum())
        for name in a:
            self.assertListEqual(a[name].applicant_id.tolist(), b[name].applicant_id.tolist())

    def test_features_do_not_include_labels_or_baseline(self):
        engineered = FeatureBuilder("engineered", True).fit_transform(self.frame.head(30))
        forbidden = {"actual_outcome", "days_to_default", "rule_based_score",
                     "rule_based_decision", "applicant_id"}
        self.assertFalse(forbidden.intersection(engineered.columns))
        self.assertTrue({"loan_to_income_ratio", "documentation_missing"}.issubset(engineered))

    def test_missing_documentation_remains_explicit(self):
        row = self.frame.loc[self.frame.documented_monthly_income.isna()].head(1)
        z = FeatureBuilder("engineered", True).fit_transform(row)
        self.assertEqual(z.documentation_missing.iloc[0], 1)
        self.assertTrue(np.isnan(z.income_discrepancy_ratio.iloc[0]))

    def test_ablation_removes_employment_only(self):
        e4 = without_employment(E2)
        self.assertFalse(e4.include_employment)
        self.assertEqual(e4.parameters, E2.parameters)
        self.assertEqual(e4.feature_set, E2.feature_set)
        z = FeatureBuilder(e4.feature_set, e4.include_employment).fit_transform(self.frame.head(10))
        self.assertNotIn("employment_status", z.columns)

    def test_threshold_selection_does_not_read_labels(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        cutoff = select_threshold_at_intervention_rate(p, target_rate=0.5)
        self.assertEqual(sum(p >= cutoff), 2)


if __name__ == "__main__":
    unittest.main()
