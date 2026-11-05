"""Evaluation and policy selection, without test-driven threshold tuning.

A positive prediction means 'intervene/review' (potential default risk), NOT
necessarily 'deny'. An actual positive is the observed `defaulted` label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def select_threshold_at_intervention_rate(probabilities: np.ndarray,
                                          target_rate: float) -> float:
    """Choose a validation-only threshold that approximates baseline flag rate.

    IMPORTANT: no outcome labels are read here. A shallow tree has tied scores,
    so an exact match to the target rate may not be possible. We report actual
    rate and use a predefined 2 pp comparison tolerance in winner selection.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite probability vector.")
    if not 0 <= target_rate <= 1:
        raise ValueError("Target intervention rate must be between 0 and 1.")
    unique = np.unique(values)
    choices = np.concatenate((unique, [np.nextafter(unique[-1], np.inf)]))
    rates = (values[:, None] >= choices[None, :]).mean(axis=0)
    errors = np.abs(rates - target_rate)
    best_choices = choices[np.isclose(errors, errors.min(), atol=1e-12)]
    # Ties: prefer the more conservative, higher threshold.
    return float(best_choices.max())


def score_binary(y_true: np.ndarray, risk_score: np.ndarray,
                 intervene: np.ndarray, name: str, split_name: str,
                 threshold: float | None, calibrated_probability: bool) -> dict:
    """Default=1. Recall measures defaults caught; FPR measures good borrowers flagged."""
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(risk_score, dtype=float)
    pred = np.asarray(intervene, dtype=int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    return {
        "split": split_name,
        "experiment": name,
        "threshold": threshold if threshold is not None else np.nan,
        "n": len(truth), "n_defaults": int(truth.sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "default_precision": float(precision_score(truth, pred, zero_division=0)),
        "default_recall": float(recall_score(truth, pred, zero_division=0)),
        "default_f1": float(f1_score(truth, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else np.nan,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else np.nan,
        "intervention_rate": float(pred.mean()),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
        # The baseline rule score is NOT a calibrated probability.
        "brier_score": (float(brier_score_loss(truth, scores))
                        if calibrated_probability else np.nan),
    }


def group_policy_metrics(rows: pd.DataFrame, y_true: np.ndarray,
                         intervene: np.ndarray, experiment: str,
                         split_name: str) -> pd.DataFrame:
    """Describe group outcomes and errors, without treating group parity as proof of fairness.

    This uses applicant employment ONLY for auditing. E4 excludes it from
    inputs to prediction, but can still audit the outputs across groups.
    """
    d = pd.DataFrame({
        "employment_status": rows["employment_status"].to_numpy(),
        "default": np.asarray(y_true, dtype=int),
        "intervene": np.asarray(intervene, dtype=int),
    })
    records = []
    for name, sub in d.groupby("employment_status", sort=True):
        actual_default = sub["default"] == 1
        real_good = ~actual_default
        flagged = sub["intervene"] == 1
        approved = ~flagged
        default_approved = int((approved & actual_default).sum())
        records.append({
            "split": split_name, "experiment": experiment,
            "employment_status": name, "n": len(sub),
            "n_defaults": int(actual_default.sum()),
            "observed_default_rate": float(actual_default.mean()),
            "approval_rate": float(approved.mean()),
            "intervention_rate": float(flagged.mean()),
            "default_recall": float((flagged & actual_default).sum() / actual_default.sum())
            if actual_default.sum() else np.nan,
            "false_positive_rate": float((flagged & real_good).sum() / real_good.sum())
            if real_good.sum() else np.nan,
            "approved_n": int(approved.sum()),
            "default_rate_among_approved": float(default_approved / approved.sum())
            if approved.sum() else np.nan,
        })
    return pd.DataFrame(records)


def choose_parent(validation_metrics: pd.DataFrame,
                  baseline_rate: float, tolerance: float = 0.02) -> str:
    """Preregistered selection: eligible rates within ±2 pp, then recall.

    If several candidates catch equal numbers of defaults at near-matched
    intervention rates, choose higher average precision, then ROC-AUC, then
    the simpler prespecified candidate order E1 → E2 → E3.
    """
    candidates = validation_metrics.loc[
        validation_metrics["experiment"].isin(["E1", "E2", "E3"])
    ].copy()
    candidates["rate_gap"] = (candidates["intervention_rate"] - baseline_rate).abs()
    eligible = candidates.loc[candidates["rate_gap"] <= tolerance].copy()
    if eligible.empty:
        raise ValueError("No candidate matches baseline intervention rate within 2 pp.")
    order = {"E1": 0, "E2": 1, "E3": 2}
    eligible["simplicity_rank"] = eligible["experiment"].map(order)
    best = eligible.sort_values(
        ["default_recall", "average_precision", "roc_auc", "simplicity_rank"],
        ascending=[False, False, False, True], kind="mergesort",
    ).iloc[0]
    return str(best["experiment"])
