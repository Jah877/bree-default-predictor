"""Application-time features and train-only preprocessing.

This module never reads the generator's latent risk/default probability or uses
Bree's existing score/action as a predictor. All statistics in imputers and
scalers are fitted inside the model Pipeline using training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_NUMERIC = (
    "stated_monthly_income", "documented_monthly_income", "loan_amount",
    "bank_ending_balance", "bank_has_overdrafts", "bank_has_consistent_deposits",
    "monthly_withdrawals", "monthly_deposits", "num_documents_submitted",
)
DERIVED_NUMERIC = (
    "loan_to_income_ratio", "income_discrepancy_ratio",
    "withdrawal_to_deposit_ratio", "documentation_missing",
)
EMPLOYMENT = "employment_status"


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Deterministically add transparent ratios; fit learns no quantities.

    ``feature_set='raw'`` uses just original application-time columns.
    ``feature_set='engineered'`` ADDS ratios and missingness indicator, allowing
    E1 versus E2 to isolate the effect of explicit feature representation.

    An income-discrepancy ratio is undefined when documentation is missing:
    preserve NaN and let the training-fitted imputer handle it. We retain an
    explicit missingness flag, rather than pretending missing income is zero.
    """

    def __init__(self, feature_set: str = "raw", include_employment: bool = True):
        self.feature_set = feature_set
        self.include_employment = include_employment

    def fit(self, X: pd.DataFrame, y=None):
        self._check_inputs(X)
        return self

    def _check_inputs(self, X: pd.DataFrame) -> None:
        if self.feature_set not in {"raw", "engineered"}:
            raise ValueError(f"Unknown feature set: {self.feature_set}")
        needed = set(RAW_NUMERIC)
        if self.include_employment:
            needed.add(EMPLOYMENT)
        absent = needed.difference(X.columns)
        if absent:
            raise ValueError(f"Missing application features: {sorted(absent)}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_inputs(X)
        numeric = list(RAW_NUMERIC)
        transformed = X.loc[:, numeric].copy()
        for name in ("bank_has_overdrafts", "bank_has_consistent_deposits"):
            transformed[name] = transformed[name].astype(float)

        if self.feature_set == "engineered":
            income = transformed["stated_monthly_income"].replace(0, np.nan)
            deposits = transformed["monthly_deposits"].replace(0, np.nan)
            documented = transformed["documented_monthly_income"]
            transformed["loan_to_income_ratio"] = transformed["loan_amount"] / income
            transformed["income_discrepancy_ratio"] = (
                (income - documented).abs() / income
            )
            transformed["withdrawal_to_deposit_ratio"] = (
                transformed["monthly_withdrawals"] / deposits
            )
            transformed["documentation_missing"] = documented.isna().astype(float)

        if self.include_employment:
            transformed[EMPLOYMENT] = X[EMPLOYMENT].astype("object")
        return transformed.replace([np.inf, -np.inf], np.nan)


def make_preprocessor(feature_set: str, include_employment: bool,
                      scale_numeric: bool) -> ColumnTransformer:
    """Build preprocessing to fit *only on the training partition*."""
    numeric_columns = list(RAW_NUMERIC)
    if feature_set == "engineered":
        numeric_columns += list(DERIVED_NUMERIC)

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    transformers = [("numeric", Pipeline(numeric_steps), numeric_columns)]
    if include_employment:
        category = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("employment", category, [EMPLOYMENT]))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
