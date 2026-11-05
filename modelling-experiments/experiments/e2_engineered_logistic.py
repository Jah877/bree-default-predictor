"""E2: controlled E1 ablation — same estimator, plus interpretable ratios."""
from specs import ExperimentSpec

SPEC = ExperimentSpec(
    key="E2", name="Logistic regression, engineered financial ratios",
    hypothesis="Relative affordability, cashflow and income mismatch improve linear prediction.",
    model_family="logistic", feature_set="engineered", include_employment=True,
    parameters={"C": 1.0, "solver": "lbfgs",
                "max_iter": 3000, "class_weight": None, "random_state": 42},
)
