"""E1: linear log-odds risk model using only original application inputs.

Why unweighted? With ~30% observed defaults, class imbalance is moderate and
we want approximately meaningful probabilities. Threshold choice is handled
separately. No hyperparameters are tuned on test outcomes.
"""
from specs import ExperimentSpec

SPEC = ExperimentSpec(
    key="E1", name="Logistic regression, raw application features",
    hypothesis="A learned additive log-odds model outperforms fixed policy weights.",
    model_family="logistic", feature_set="raw", include_employment=True,
    parameters={"C": 1.0, "solver": "lbfgs",
                "max_iter": 3000, "class_weight": None, "random_state": 42},
)
