"""E3: interpretable shallow CART tree (fixed complexity, no hidden tuning).

Provides nonlinear splits and feature interactions, unlike additive logits.
Conservative depth/leaf settings limit variance for this 1,101-row training set.
"""
from specs import ExperimentSpec

SPEC = ExperimentSpec(
    key="E3", name="Shallow decision tree, engineered financial ratios",
    hypothesis="Learned split thresholds and interactions outperform additive log-odds.",
    model_family="tree", feature_set="engineered", include_employment=True,
    parameters={"max_depth": 4, "min_samples_leaf": 30,
                "min_samples_split": 60, "criterion": "gini",
                "class_weight": None, "random_state": 42},
)
