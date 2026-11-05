"""Exploratory analysis of Bree's synthetic loan dataset.

Run from any working directory:
    python data-analysis/analysis.py

Policy diagnostics, distributions, feature associations, group fairness, and
one-at-a-time policy sensitivity. All outcome-dependent diagnostics use ONLY
training data to preserve validation/test as untouched evaluation partitions.
No machine-learning model is fitted by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from baseline import WEIGHTS, compare_to_stored, policy_action, score_components
from splits import make_splits


ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    """Load the project's single authoritative configuration file."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_and_load(settings: dict) -> pd.DataFrame:
    """Read generated CSV, verify schema and key policy invariants."""
    dataset_path = ROOT / settings["paths"]["dataset"]
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Missing {dataset_path}. First run: python data-analysis/run_generation.py"
        )
    frame = pd.read_csv(dataset_path)
    required = {
        "applicant_id", "stated_monthly_income", "documented_monthly_income",
        "loan_amount", "employment_status", "bank_ending_balance",
        "bank_has_overdrafts", "bank_has_consistent_deposits", "monthly_withdrawals",
        "monthly_deposits", "num_documents_submitted", "rule_based_score",
        "rule_based_decision", "actual_outcome", "days_to_default",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing expected dataset columns: {sorted(missing)}")
    if frame["applicant_id"].duplicated().any():
        raise ValueError("Duplicate applicant_id values found.")
    allowed_outcomes = {"repaid", "defaulted", "ongoing"}
    allowed_actions = {"approved", "denied", "flagged_for_review"}
    if not frame["actual_outcome"].isin(allowed_outcomes).all():
        raise ValueError("Unexpected outcome label(s) found.")
    if not frame["rule_based_decision"].isin(allowed_actions).all():
        raise ValueError("Unexpected baseline action(s) found.")
    for field in ("bank_has_overdrafts", "bank_has_consistent_deposits"):
        if frame[field].isna().any():
            raise ValueError(f"Boolean column {field} contains missing entries.")
    if frame["rule_based_score"].lt(0).any() or frame["rule_based_score"].gt(100).any():
        raise ValueError("Rule-based scores must be between 0 and 100.")
    return frame


def write_data_quality(frame: pd.DataFrame, output: Path) -> None:
    """Export distributions and null counts, without treating ongoing as negatives."""
    quality = pd.DataFrame({
        "dtype": frame.dtypes.astype(str),
        "missing_count": frame.isna().sum(),
        "missing_pct": (100 * frame.isna().mean()).round(2),
        "unique_count": frame.nunique(dropna=True),
    })
    quality.to_csv(output / "data_quality.csv", index_label="feature")
    frame["actual_outcome"].value_counts().rename_axis("outcome").to_csv(
        output / "outcome_counts.csv", header=["count"]
    )
    frame["rule_based_decision"].value_counts().rename_axis("decision").to_csv(
        output / "baseline_decision_counts.csv", header=["count"]
    )


def with_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create transparent ratios for exploration, not a fitted preprocessing pipeline."""
    derived = frame.copy()
    income = derived["stated_monthly_income"].replace(0, np.nan)
    documented = derived["documented_monthly_income"]
    deposits = derived["monthly_deposits"].replace(0, np.nan)
    derived["loan_to_income_ratio"] = derived["loan_amount"] / income
    derived["withdrawal_to_deposit_ratio"] = derived["monthly_withdrawals"] / deposits
    derived["income_discrepancy_ratio"] = (income - documented).abs() / income
    derived["has_documented_income"] = documented.notna().astype(int)
    return derived


def analyze_associations(train: pd.DataFrame, output: Path, bins: int) -> None:
    """Explore association and redundancy, without claiming causal importance."""
    derived = with_features(train)
    numeric = derived.select_dtypes(include=["number"]).drop(
        columns=["rule_based_score", "days_to_default"], errors="ignore"
    )
    # Numeric-input correlations; exclude prior policy and outcome-derived values.
    corr = numeric.corr(method="spearman")
    corr.to_csv(output / "spearman_feature_correlations.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)), labels=corr.columns,
                  rotation=70, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(corr.index)), labels=corr.index, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman correlation")
    ax.set_title("Training-set numeric feature dependencies")
    fig.tight_layout()
    fig.savefig(output / "figures" / "feature_dependencies.png", dpi=160)
    plt.close(fig)

    derived["is_default"] = derived["actual_outcome"].eq("defaulted").astype(int)
    bucket = pd.qcut(derived["loan_to_income_ratio"], q=bins, duplicates="drop")
    grouped = derived.groupby(bucket, observed=True).agg(
        n=("is_default", "size"), default_rate=("is_default", "mean"),
        min_ratio=("loan_to_income_ratio", "min"),
        max_ratio=("loan_to_income_ratio", "max"),
    )
    grouped.to_csv(output / "loan_burden_default_rate_train.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(grouped))
    ax.bar(positions, 100 * grouped["default_rate"].to_numpy())
    ax.set_xticks(positions, labels=[f"{lo:.2f}–{hi:.2f}" for lo, hi in
                                    zip(grouped.min_ratio, grouped.max_ratio)],
                  rotation=35, ha="right")
    ax.set_ylabel("Observed default rate (%)")
    ax.set_xlabel("Loan amount / stated monthly income (bin range)")
    ax.set_title("Training-set default rate by loan burden")
    for x, pct, n in zip(positions, grouped.default_rate, grouped.n):
        ax.annotate(f"n={n}", (x, 100 * pct), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figures" / "loan_burden_default_rate.png", dpi=160)
    plt.close(fig)



def descriptive_risk_signals_train(train: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Describe observed default rates for understandable binary financial signals.

    These are *univariate associations*, not adjusted feature importance or
    causal effects. Signals are generated only from training-set features.
    """
    d = with_features(train)
    default = d["actual_outcome"].eq("defaulted")
    signals = {
        "missing_documented_income": d["documented_monthly_income"].isna(),
        "income_gap_over_10pct": d["income_discrepancy_ratio"] > 0.10,
        "loan_over_one_third_income": d["loan_to_income_ratio"] > (1 / 3),
        "bank_overdrafts": d["bank_has_overdrafts"].astype(bool),
        "balance_at_or_below_500": d["bank_ending_balance"] <= 500,
        "withdrawals_over_80pct_deposits": d["withdrawal_to_deposit_ratio"] > 0.80,
        "inconsistent_deposits": ~d["bank_has_consistent_deposits"].astype(bool),
    }
    rows = []
    for name, mask in signals.items():
        n_pos, n_neg = int(mask.sum()), int((~mask).sum())
        rate_pos = float(default[mask].mean()) if n_pos else np.nan
        rate_neg = float(default[~mask].mean()) if n_neg else np.nan
        rows.append({
            "signal": name,
            "n_with_signal": n_pos,
            "n_without_signal": n_neg,
            "default_rate_with_signal": rate_pos,
            "default_rate_without_signal": rate_neg,
            "percentage_point_difference": 100 * (rate_pos - rate_neg),
        })
    results = pd.DataFrame(rows).sort_values(
        "percentage_point_difference", ascending=False
    )
    results.to_csv(output / "descriptive_risk_signals_train.csv", index=False)
    return results


def evaluate_baseline_train(train: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Diagnostic results on TRAIN ONLY. Final comparison must use held-out test.

    y=1 means defaulted. For strict mapping, 'denied' predicts default;
    for intervention mapping, 'denied' and 'flagged_for_review' predict default.
    Both use the *actual* original rule decisions. ROC-AUC uses -rule score.
    """
    y = train["actual_outcome"].eq("defaulted").astype(int)
    auc = roc_auc_score(y, -train["rule_based_score"])
    rows = []
    for mapping, actions in {
        "strict_denial": ["denied"],
        "intervention_required": ["denied", "flagged_for_review"],
    }.items():
        pred = train["rule_based_decision"].isin(actions).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            "mapping": mapping,
            "n": int(len(y)),
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
            "default_precision": precision_score(y, pred, zero_division=0),
            "default_recall": recall_score(y, pred, zero_division=0),
            "default_f1": f1_score(y, pred, zero_division=0),
            "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
            "false_negative_rate": fn / (fn + tp) if fn + tp else np.nan,
            "roc_auc_from_score": auc,
        })
    table = pd.DataFrame(rows)
    table.to_csv(output / "baseline_metrics_train_only.csv", index=False)
    return table


def employment_fairness_train(train: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Group approval/default statistics on observed outcomes, training split.

    'Group default rate' is based on all group members with observed labels.
    'Default rate if approved' conditions additionally on historical approval;
    they answer different questions and must not be interchanged.
    """
    rows = []
    for name, group in train.groupby("employment_status", sort=True):
        approved = group["rule_based_decision"].eq("approved")
        defaulted = group["actual_outcome"].eq("defaulted")
        rows.append({
            "employment_status": name,
            "n": len(group),
            "baseline_approval_rate": approved.mean(),
            "observed_group_default_rate": defaulted.mean(),
            "default_rate_among_baseline_approved": defaulted[approved].mean()
                if approved.any() else np.nan,
            "baseline_review_rate": group["rule_based_decision"].eq("flagged_for_review").mean(),
            "baseline_denial_rate": group["rule_based_decision"].eq("denied").mean(),
        })
    table = pd.DataFrame(rows)
    table.to_csv(output / "employment_fairness_train.csv", index=False)
    positions = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(positions - 0.18, 100 * table["baseline_approval_rate"], 0.36,
           label="Baseline approved")
    ax.bar(positions + 0.18, 100 * table["observed_group_default_rate"], 0.36,
           label="Observed group default")
    ax.set_xticks(positions, labels=table["employment_status"])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Training-set approval and outcome rates by employment status")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figures" / "employment_fairness.png", dpi=160)
    plt.close(fig)
    return table


def apply_scenario(original: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Vary exactly one observed feature; does not simulate a causal applicant."""
    field, op, value = spec["feature"], spec["operation"], spec["value"]
    if field not in original.columns:
        raise KeyError(f"Unknown sensitivity feature: {field}")
    if field not in {
        "loan_amount", "stated_monthly_income", "documented_monthly_income",
        "bank_ending_balance", "bank_has_overdrafts",
        "bank_has_consistent_deposits", "employment_status",
        "monthly_deposits", "monthly_withdrawals",
    }:
        raise ValueError(f"Feature not eligible for sensitivity test: {field}")
    modified = original.copy()
    if op == "multiply":
        modified[field] = modified[field] * value
    elif op == "add":
        modified[field] = modified[field] + value
    elif op == "set":
        modified[field] = value
    else:
        raise ValueError(f"Unrecognized operation: {op}")
    if field in {"loan_amount", "stated_monthly_income", "documented_monthly_income",
                 "bank_ending_balance", "monthly_deposits", "monthly_withdrawals"}:
        modified[field] = modified[field].clip(lower=0)
    return modified


def sensitivity_analysis(train: pd.DataFrame, settings: dict, output: Path) -> pd.DataFrame:
    """One-factor-at-a-time baseline stress tests (not estimated feature effects).

    All delta scores compare reconstructed baseline before/after perturbation,
    NOT stored scores to reconstructed scores; this avoids rounding mismatch.
    """
    baseline_score = score_components(train)["reconstructed_score"]
    threshold = settings["baseline"]
    baseline_action = policy_action(baseline_score,
        threshold["approve_at_or_above"], threshold["review_at_or_above"])
    rows = []
    for spec in settings["analysis"]["sensitivity_scenarios"]:
        # When specified, restrict a scenario to meaningful source-state rows.
        # For example, employed -> self-employed, not every group -> self-employed.
        condition = spec.get("only_when")
        if condition:
            eligible = train.loc[train[condition["feature"]].eq(condition["equals"])]
        else:
            eligible = train
        if eligible.empty:
            continue
        counterfactual = apply_scenario(eligible, spec)
        old_score = baseline_score.loc[eligible.index]
        old_action = baseline_action.loc[eligible.index]
        new_score = score_components(counterfactual)["reconstructed_score"]
        new_action = policy_action(new_score,
            threshold["approve_at_or_above"], threshold["review_at_or_above"])
        was_approved = old_action.eq("approved")
        now_approved = new_action.eq("approved")
        rows.append({
            "scenario": spec["name"],
            "feature": spec["feature"],
            "n": len(eligible),
            "input_rows_changed": int(eligible[spec["feature"]].ne(counterfactual[spec["feature"]]).sum()),
            "mean_score_change": float((new_score - old_score).mean()),
            "mean_abs_score_change": float((new_score - old_score).abs().mean()),
            "fraction_action_changed": float(new_action.ne(old_action).mean()),
            "approved_to_not_approved": int((was_approved & ~now_approved).sum()),
            "not_approved_to_approved": int((~was_approved & now_approved).sum()),
        })
    table = pd.DataFrame(rows)
    table.to_csv(output / "baseline_sensitivity_train.csv", index=False)
    return table


def export_baseline_components(train: pd.DataFrame, output: Path) -> None:
    """Inspect component contributions without confusing rules with predictive value."""
    comp = score_components(train)
    summary = pd.DataFrame([
        {
            "component": name,
            "policy_weight": weight,
            "mean_contribution": comp[f"{name}_weighted"].mean(),
            "std_contribution": comp[f"{name}_weighted"].std(),
        }
        for name, weight in WEIGHTS.items()
    ])
    summary.to_csv(output / "baseline_component_summary_train.csv", index=False)


def main() -> None:
    settings = load_config()
    frame = validate_and_load(settings)
    out = ROOT / settings["paths"]["outputs"]
    (out / "figures").mkdir(parents=True, exist_ok=True)
    write_data_quality(frame, out)
    splits = make_splits(frame, settings)
    training = splits["train"]
    print(f"Dataset rows: {len(frame)} | outcome counts: "
          f"{frame.actual_outcome.value_counts().to_dict()}")
    print(f"Known-label splits: "
          f"{ {name: len(part) for name, part in splits.items()} }")

    components = score_components(frame)
    baseline_settings = settings["baseline"]
    check = compare_to_stored(frame, components,
        baseline_settings["approve_at_or_above"],
        baseline_settings["review_at_or_above"])
    with (out / "baseline_reconstruction_check.json").open("w", encoding="utf-8") as file:
        json.dump(check, file, indent=2)
    print("Reconstructed baseline vs stored baseline:", check)
    if check["policy_decision_matches"] < len(frame):
        print("WARNING: reconstructed policy does not exactly match all stored decisions; "
              "use original columns for historical baseline metrics.")

    export_baseline_components(training, out)
    analyze_associations(training, out, int(settings["analysis"]["quantile_bins"]))
    risk_signals = descriptive_risk_signals_train(training, out)
    metrics = evaluate_baseline_train(training, out)
    fairness = employment_fairness_train(training, out)
    sensitivity = sensitivity_analysis(training, settings, out)
    print("\nTraining-set descriptive risk signals (percentage-point differences):\n",
          risk_signals[["signal", "n_with_signal", "percentage_point_difference"]].round(2).to_string(index=False))
    print("\nBaseline TRAIN diagnostics (not final test results):\n",
          metrics.round(3).to_string(index=False))
    print("\nEmployment-group TRAIN diagnostics:\n",
          fairness.round(3).to_string(index=False))
    print("\nOne-feature-at-a-time baseline sensitivity (TRAIN):\n",
          sensitivity.round(3).to_string(index=False))
    print(f"\nArtifacts saved under: {out}")
    print("Held-out validation/test outcomes are intentionally not analyzed here.")


if __name__ == "__main__":
    main()
