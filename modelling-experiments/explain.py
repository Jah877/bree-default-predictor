"""Model diagnostics; all explanations are predictive, not causal.

Coefficient sizes are conditional on included correlated variables. Tree
impurity importance can favour features with many split opportunities.
Permutation importance estimates loss of validation ROC-AUC when an ORIGINAL
input is shuffled (including any engineered features derived from it).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def export_importance(model, spec, X_val: pd.DataFrame, y_val: np.ndarray,
                      path: Path, seed: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    transformed = model.named_steps["features"].transform(X_val)
    prep = model.named_steps["preprocessing"]
    feature_names = prep.get_feature_names_out()
    clf = model.named_steps["classifier"]
    if spec.model_family == "logistic":
        pd.DataFrame({
            "transformed_feature": feature_names,
            "standardized_or_onehot_coefficient": clf.coef_[0],
            "odds_ratio_per_transformed_unit": np.exp(clf.coef_[0]),
        }).sort_values("standardized_or_onehot_coefficient", key=abs,
                      ascending=False).to_csv(path / "coefficients.csv", index=False)
    else:
        pd.DataFrame({
            "transformed_feature": feature_names,
            "impurity_importance": clf.feature_importances_,
        }).sort_values("impurity_importance", ascending=False).to_csv(
            path / "tree_importance.csv", index=False)

    # Important: only applicant-time columns, and only validation targets.
    # Permute unengineered inputs, so derived ratios vary consistently with them.
    permutation_inputs = X_val.drop(columns=["applicant_id"], errors="ignore")
    p = permutation_importance(model, permutation_inputs, y_val, scoring="roc_auc",
                               n_repeats=10, random_state=seed)
    pd.DataFrame({
        "original_input": permutation_inputs.columns,
        "validation_auc_decrease_mean": p.importances_mean,
        "validation_auc_decrease_std": p.importances_std,
    }).sort_values("validation_auc_decrease_mean", ascending=False).to_csv(
        path / "permutation_importance.csv", index=False)


def local_explanations(model, spec, X: pd.DataFrame, actual_y: np.ndarray,
                       risk: np.ndarray, intervene: np.ndarray,
                       n_examples: int = 4) -> pd.DataFrame:
    """Give auditors concrete examples from validation ONLY.

    Logistic: largest additive log-odds contributions for each example.
    Tree: exact split decisions on the prediction path.
    Examples intentionally include both correct and incorrect interventions.
    """
    candidates = []
    truth = np.asarray(actual_y)
    flags = np.asarray(intervene)
    for condition in [((flags == 1) & (truth == 1)),
                      ((flags == 1) & (truth == 0)),
                      ((flags == 0) & (truth == 1)),
                      ((flags == 0) & (truth == 0))]:
        idx = np.flatnonzero(condition)
        if len(idx):
            candidates.append(int(idx[0]))
    candidates = candidates[:n_examples]

    changed = model.named_steps["features"].transform(X.iloc[candidates])
    prep = model.named_steps["preprocessing"]
    transformed = prep.transform(changed)
    names = prep.get_feature_names_out()
    clf = model.named_steps["classifier"]
    records = []
    for offset, idx in enumerate(candidates):
        values = np.asarray(transformed[offset]).ravel()
        if spec.model_family == "logistic":
            contributions = values * clf.coef_[0]
            top = np.argsort(-np.abs(contributions))[:5]
            explanation = [
                {"feature": str(names[k]), "log_odds_contribution": round(float(contributions[k]), 4)}
                for k in top
            ]
            intercept = float(clf.intercept_[0])
        else:
            nodes = clf.decision_path(values.reshape(1, -1)).indices
            path_steps = []
            for node in nodes:
                index = int(clf.tree_.feature[node])
                if index < 0:  # leaf
                    continue
                cutoff = float(clf.tree_.threshold[node])
                current = float(values[index])
                path_steps.append(
                    f"{names[index]} {'<=' if current <= cutoff else '>'} {cutoff:.4g}"
                )
            explanation = path_steps
            intercept = np.nan
        records.append({
            "applicant_id": str(X.iloc[idx]["applicant_id"]),
            "actual_default": int(truth[idx]),
            "intervene": int(flags[idx]),
            "predicted_default_probability": float(risk[idx]),
            "model": spec.key,
            "explanation_type": "log_odds_contributions" if spec.model_family == "logistic" else "tree_path",
            "logistic_intercept": intercept,
            "details_json": json.dumps(explanation),
        })
    return pd.DataFrame(records)
