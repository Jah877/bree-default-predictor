"""Reconstruct the supplied rule-based policy from *observable CSV columns*.

Bree computes its stored score before rounding deposits/withdrawals and has an
internal misrepresentation flag absent from the exported table. We infer that
flag via the generator's non-overlapping documented/stated income ranges.
Consequently tiny numerical score discrepancies may remain; the CSV policy
columns are authoritative for retrospective evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


WEIGHTS = {
    "income_verification": 0.30,
    "income_level": 0.25,
    "account_stability": 0.20,
    "employment": 0.15,
    "cashflow": 0.10,
}


def score_components(applicants: pd.DataFrame) -> pd.DataFrame:
    """Return raw component scores (0–100), their contributions and total score.

    Do not substitute these scores for the original `rule_based_score` when
    retrospectively evaluating Bree's historic decisions.
    """
    stated = applicants["stated_monthly_income"].astype(float)
    documented = applicants["documented_monthly_income"].astype(float)
    loan = applicants["loan_amount"].astype(float)
    balance = applicants["bank_ending_balance"].astype(float)
    deposits = applicants["monthly_deposits"].astype(float)
    withdrawals = applicants["monthly_withdrawals"].astype(float)

    has_docs = documented.notna()
    relative_discrepancy = (stated - documented).abs() / stated.replace(0, np.nan)
    # Only possible in Bree's misrepresentation branch, given the supplied ranges.
    apparent_misrepresentation = has_docs & (documented / stated < 0.5)
    verification = np.select(
        [~has_docs, apparent_misrepresentation, relative_discrepancy <= 0.1],
        [0, 0, 95],
        default=40,
    )
    income_level = np.select(
        [stated >= 3 * loan, stated >= 2 * loan],
        [90, 45],
        default=0,
    )
    account = np.where(balance > 500, 40, 10)
    account = account + np.where(~applicants["bank_has_overdrafts"].astype(bool), 30, 0)
    account = account + np.where(applicants["bank_has_consistent_deposits"].astype(bool), 30, 0)
    account = np.clip(account, 0, 100)
    employment = applicants["employment_status"].map(
        {"employed": 100, "self_employed": 60, "unemployed": 20}
    )
    if employment.isna().any():
        raise ValueError("Unrecognized employment_status encountered.")
    with np.errstate(divide="ignore", invalid="ignore"):
        cashflow = np.where(
            deposits > 0,
            np.clip(100 - 100 * withdrawals / deposits, 0, 100),
            50,
        )

    raw = pd.DataFrame(
        {
            "income_verification": verification,
            "income_level": income_level,
            "account_stability": account,
            "employment": employment.to_numpy(),
            "cashflow": cashflow,
        },
        index=applicants.index,
    )
    result = raw.add_suffix("_raw")
    for name, weight in WEIGHTS.items():
        result[f"{name}_weighted"] = raw[name] * weight
    result["reconstructed_score"] = result[
        [f"{name}_weighted" for name in WEIGHTS]
    ].sum(axis=1).round(1)
    return result


def policy_action(scores: pd.Series, approve_at: float = 75,
                  review_at: float = 50) -> pd.Series:
    """Map loan-review score to the original three actions."""
    return pd.Series(
        np.where(scores >= approve_at, "approved",
                 np.where(scores >= review_at, "flagged_for_review", "denied")),
        index=scores.index,
        name="policy_action",
    )


def compare_to_stored(applicants: pd.DataFrame, components: pd.DataFrame,
                      approve_at: float = 75, review_at: float = 50) -> dict:
    """Diagnostic only: compare reconstruction with authoritative stored values."""
    diff = components["reconstructed_score"] - applicants["rule_based_score"]
    reconstructed_actions = policy_action(components["reconstructed_score"],
                                          approve_at, review_at)
    return {
        "rows": len(applicants),
        "exact_score_matches": int(diff.eq(0).sum()),
        "score_mean_absolute_difference": float(diff.abs().mean()),
        "score_max_absolute_difference": float(diff.abs().max()),
        "policy_decision_matches": int(
            reconstructed_actions.eq(applicants["rule_based_decision"]).sum()
        ),
    }
