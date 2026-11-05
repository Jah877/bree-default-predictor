"""Run E0–E4, select a parent using validation, and optionally unlock test once.

From the repository root:
    python modelling-experiments/run_experiments.py
    python modelling-experiments/run_experiments.py --finalize-test

No test labels or metrics are accessed in ordinary validation-only mode. In
final mode, model settings and thresholds have already been frozen by
validation; the held-out test is used ONLY for final reporting. Refrain from
revising a model after seeing these test results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "data-analysis"))
from analysis import validate_and_load  # Uses root config, no model evaluation.
from splits import make_splits
from evaluation import (
    choose_parent, group_policy_metrics, score_binary,
    select_threshold_at_intervention_rate,
)
from explain import export_importance, local_explanations
from specs import ExperimentSpec, make_pipeline
from features import RAW_NUMERIC, EMPLOYMENT
from experiments.e0_rule_baseline import ACTION_SETS
from experiments.e1_raw_logistic import SPEC as E1
from experiments.e2_engineered_logistic import SPEC as E2
from experiments.e3_shallow_tree import SPEC as E3
from experiments.e4_employment_ablation import make_ablation

with (ROOT / "config.yaml").open("r", encoding="utf-8") as _h:
    SETTINGS = yaml.safe_load(_h)
OUT = ROOT / SETTINGS["paths"]["modelling_outputs"]
FEATURE_INPUTS = ["applicant_id", *RAW_NUMERIC, EMPLOYMENT]


def labels(frame: pd.DataFrame, settings: dict) -> np.ndarray:
    return frame[settings["data"]["target"]].eq(
        settings["data"]["positive_label"]
    ).astype(int).to_numpy()


def get_X(frame: pd.DataFrame) -> pd.DataFrame:
    """Hard allowlist prevents outcome, days-to-default and rule-score leakage."""
    return frame.loc[:, FEATURE_INPUTS].copy()


def baseline_evaluation(rows: pd.DataFrame, y: np.ndarray,
                        split_name: str) -> tuple[list[dict], list[pd.DataFrame]]:
    score = 1 - rows["rule_based_score"].astype(float).to_numpy() / 100.0
    rows_metrics, groups = [], []
    for variant, interventions in ACTION_SETS.items():
        intervention = rows["rule_based_decision"].isin(interventions).astype(int).to_numpy()
        rows_metrics.append(score_binary(
            y, score, intervention, variant, split_name,
            threshold=None, calibrated_probability=False,
        ))
        groups.append(group_policy_metrics(rows, y, intervention, variant, split_name))
    return rows_metrics, groups


def evaluate_model(model, key: str, rows: pd.DataFrame, y: np.ndarray,
                   threshold: float, split_name: str) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(get_X(rows))[:, 1]
    intervention = (probabilities >= threshold).astype(int)
    metrics = score_binary(y, probabilities, intervention, key, split_name,
                           threshold=threshold, calibrated_probability=True)
    fairness = group_policy_metrics(rows, y, intervention, key, split_name)
    return metrics, fairness, probabilities, intervention


def make_prediction_table(rows: pd.DataFrame, y: np.ndarray,
                          predictions: dict) -> pd.DataFrame:
    report = rows[["applicant_id", "employment_status", "rule_based_score",
                   "rule_based_decision"]].reset_index(drop=True).copy()
    report["actual_default"] = y
    for key, (probabilities, intervention) in predictions.items():
        report[f"{key}_predicted_default_probability"] = probabilities
        report[f"{key}_intervene"] = intervention
    return report


def comparison_report(metrics: pd.DataFrame) -> pd.DataFrame:
    """Absolute counts make the lending consequences tangible."""
    baseline = metrics.loc[metrics.experiment == "E0_intervention"].iloc[0]
    result = metrics.copy()
    result["additional_defaults_caught_vs_E0"] = result["tp"] - int(baseline.tp)
    result["additional_good_applicants_flagged_vs_E0"] = result["fp"] - int(baseline.fp)
    result["intervention_rate_gap_vs_E0_pp"] = (
        result["intervention_rate"] - float(baseline.intervention_rate)
    ) * 100
    return result


def plot_validation_curves(y: np.ndarray, scores: dict[str, np.ndarray]) -> None:
    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.7, 5.3))
    for key, risk in scores.items():
        RocCurveDisplay.from_predictions(y, risk, name=key, ax=ax)
    ax.set_title("Validation ROC: baseline ranking vs learned models")
    fig.tight_layout()
    fig.savefig(figs / "validation_roc.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.7, 5.3))
    for key, risk in scores.items():
        PrecisionRecallDisplay.from_predictions(y, risk, name=key, ax=ax)
    ax.set_title("Validation precision–recall (defaulted = positive)")
    fig.tight_layout()
    fig.savefig(figs / "validation_precision_recall.png", dpi=150)
    plt.close(fig)


def plot_confusion(y: np.ndarray, predictions: dict[str, np.ndarray],
                   prefix: str) -> None:
    directory = OUT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    for key, intervention in predictions.items():
        fig, ax = plt.subplots(figsize=(5.1, 4.2))
        ConfusionMatrixDisplay.from_predictions(
            y, intervention, labels=[0, 1], display_labels=["Repaid", "Defaulted"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_xlabel("Predicted intervention (0=approve, 1=intervene)")
        ax.set_ylabel("Observed outcome")
        ax.set_title(f"{prefix}: {key}")
        fig.tight_layout()
        fig.savefig(directory / f"{prefix}_{key}_confusion.png", dpi=150)
        plt.close(fig)


def main(finalize_test: bool = False, overwrite_test: bool = False) -> None:
    settings = SETTINGS
    OUT.mkdir(parents=True, exist_ok=True)
    if finalize_test and (OUT / "test_metrics.csv").exists() and not overwrite_test:
        raise FileExistsError(
            "Test evaluation already exists. Do not repeatedly select models from it. "
            "For a deliberate reproducibility rerun only, pass --overwrite-test."
        )

    frame = validate_and_load(settings)
    parts = make_splits(frame, settings)
    train, val = parts["train"], parts["validation"]
    y_train, y_val = labels(train, settings), labels(val, settings)
    if len(set(y_train)) != 2 or len(set(y_val)) != 2:
        raise RuntimeError("Both classes must be present in train and validation.")

    primary_intervention = val["rule_based_decision"].ne("approved").to_numpy(dtype=int)
    target_rate = float(primary_intervention.mean())
    baseline_scores = 1 - val["rule_based_score"].to_numpy(dtype=float) / 100
    base_metrics, base_groups = baseline_evaluation(val, y_val, "validation")

    specs: dict[str, ExperimentSpec] = {s.key: s for s in (E1, E2, E3)}
    fitted = {}
    thresholds = {}
    all_metrics = list(base_metrics)
    all_groups = list(base_groups)
    val_predictions = {}
    val_curve_scores = {"E0 (rule score)": baseline_scores}
    val_interventions = {"E0_intervention": primary_intervention}

    for key, spec in specs.items():
        model = make_pipeline(spec)
        model.fit(get_X(train), y_train)
        p = model.predict_proba(get_X(val))[:, 1]
        threshold = select_threshold_at_intervention_rate(p, target_rate)
        metrics, groups, _, flagged = evaluate_model(
            model, key, val, y_val, threshold, "validation"
        )
        fitted[key] = model
        thresholds[key] = threshold
        all_metrics.append(metrics)
        all_groups.append(groups)
        val_predictions[key] = (p, flagged)
        val_curve_scores[key] = p
        val_interventions[key] = flagged

    table_for_selection = pd.DataFrame(all_metrics)
    allowed_rate_gap = float(settings["modelling"]["selected_parent_max_intervention_gap"])
    parent_key = choose_parent(table_for_selection, target_rate,
                               tolerance=allowed_rate_gap)
    e4 = make_ablation(specs[parent_key])
    specs[e4.key] = e4
    ablated = make_pipeline(e4)
    ablated.fit(get_X(train), y_train)
    p_e4 = ablated.predict_proba(get_X(val))[:, 1]
    t_e4 = select_threshold_at_intervention_rate(p_e4, target_rate)
    metrics_e4, groups_e4, _, flags_e4 = evaluate_model(
        ablated, "E4", val, y_val, t_e4, "validation"
    )
    fitted["E4"] = ablated
    thresholds["E4"] = t_e4
    all_metrics.append(metrics_e4)
    all_groups.append(groups_e4)
    val_predictions["E4"] = (p_e4, flags_e4)
    val_curve_scores["E4"] = p_e4
    val_interventions["E4"] = flags_e4

    validation = comparison_report(pd.DataFrame(all_metrics))
    validation.to_csv(OUT / "validation_metrics.csv", index=False)
    pd.concat(all_groups, ignore_index=True).to_csv(
        OUT / "validation_employment_fairness.csv", index=False
    )
    make_prediction_table(val, y_val, val_predictions).to_csv(
        OUT / "validation_predictions.csv", index=False
    )
    plot_validation_curves(y_val, val_curve_scores)
    plot_confusion(y_val, {
        "E0_intervention": primary_intervention,
        parent_key: val_interventions[parent_key],
        "E4": val_interventions["E4"],
    }, "validation")

    # Feature explanation from held-out VALIDATION samples; not test samples.
    seed = int(settings["project"]["random_state"])
    for key in ["E1", "E2", "E3", "E4"]:
        export_importance(fitted[key], specs[key], get_X(val), y_val,
                          OUT / "importance" / key, seed)
    local_explanations(
        fitted[parent_key], specs[parent_key], get_X(val), y_val,
        val_predictions[parent_key][0], val_predictions[parent_key][1],
    ).to_csv(OUT / "example_decisions_validation.csv", index=False)

    dataset_path = ROOT / settings["paths"]["dataset"]
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest = {
        "dataset_sha256": dataset_hash,
        "split_sizes": {name: len(rows) for name, rows in parts.items()},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation",
        "selected_parent": parent_key,
        "e4_ablates": parent_key,
        "chosen_model": parent_key,
        "e4_role": "diagnostic; not automatically promoted based on its test score",
        "baseline_primary_definition": "review OR deny = intervene; approve = no intervention",
        "baseline_strict_secondary_definition": "deny only = predicted default",
        "threshold_strategy": "closest possible to baseline validation intervention rate, no labels read for threshold",
        "target_validation_intervention_rate": target_rate,
        "winner_eligibility_rate_gap_at_most": allowed_rate_gap,
        "selection_rank": ["default_recall", "average_precision", "roc_auc", "prespecified_simplicity"],
        "test_unlocked": bool(finalize_test),
        "test_not_used_for_model_or_threshold_selection": True,
        "experiments": {k: asdict(v) for k, v in specs.items()},
        "thresholds": thresholds,
        "limitation": "Observational synthetic labels, not causal effects or deployment-ready approval policy.",
    }
    with (OUT / "selection_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Validation baseline intervention rate: {target_rate:.1%}")
    print(f"Validation-selected parent: {parent_key}; E4 removes employment from {parent_key}")
    print(validation[["experiment", "default_recall", "false_positive_rate",
                      "intervention_rate", "roc_auc", "average_precision", "brier_score"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Validation reports: {OUT}")
    if not finalize_test:
        print("Test data remain unexamined. Use --finalize-test once when satisfied.")
        return

    # TEST LOCK: No fitting, threshold choice, or model selection in this block.
    test = parts["test"]
    y_test = labels(test, settings)
    test_metrics, test_groups = baseline_evaluation(test, y_test, "test")
    test_predictions = {}
    confusions = {
        "E0_intervention": test["rule_based_decision"].ne("approved").astype(int).to_numpy()
    }
    for key in ["E1", "E2", "E3", "E4"]:
        metric, groups, p, intervention = evaluate_model(
            fitted[key], key, test, y_test, thresholds[key], "test"
        )
        test_metrics.append(metric)
        test_groups.append(groups)
        test_predictions[key] = (p, intervention)
        if key in (parent_key, "E4"):
            confusions[key] = intervention
    comparison_report(pd.DataFrame(test_metrics)).to_csv(
        OUT / "test_metrics.csv", index=False
    )
    pd.concat(test_groups, ignore_index=True).to_csv(
        OUT / "test_employment_fairness.csv", index=False
    )
    make_prediction_table(test, y_test, test_predictions).to_csv(
        OUT / "test_predictions.csv", index=False
    )
    plot_confusion(y_test, confusions, "test")
    print("FINAL TEST REPORT WRITTEN. Do not adapt the model or decision threshold based on it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-test", action="store_true",
                        help="Evaluate locked models against held-out test once")
    parser.add_argument("--overwrite-test", action="store_true",
                        help="Only for exact reproducibility reruns, not model selection")
    args = parser.parse_args()
    if args.overwrite_test and not args.finalize_test:
        parser.error("--overwrite-test requires --finalize-test")
    main(finalize_test=args.finalize_test, overwrite_test=args.overwrite_test)
