"""Parameter ownership: each E1/E2/E3 specification lives in its own file."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from features import FeatureBuilder, make_preprocessor


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    name: str
    hypothesis: str
    model_family: str
    feature_set: str
    include_employment: bool
    parameters: dict[str, Any]


def make_pipeline(spec: ExperimentSpec):
    """Fit feature engineering, imputation, and classifier as one ML pipeline."""
    if spec.model_family == "logistic":
        estimator = LogisticRegression(**spec.parameters)
        scale_numeric = True
    elif spec.model_family == "tree":
        estimator = DecisionTreeClassifier(**spec.parameters)
        scale_numeric = False
    else:
        raise ValueError(f"Unsupported estimator: {spec.model_family}")
    return Pipeline([
        ("features", FeatureBuilder(spec.feature_set, spec.include_employment)),
        ("preprocessing", make_preprocessor(
            spec.feature_set, spec.include_employment, scale_numeric
        )),
        ("classifier", estimator),
    ])


def without_employment(parent: ExperimentSpec) -> ExperimentSpec:
    """E4 changes only one design choice, keeping all other settings fixed."""
    return replace(
        parent, key="E4", name=f"Employment ablation of {parent.key}",
        include_employment=False,
        hypothesis="Does removal of employment status alter risk prediction or group disparities?",
    )
