# Bree — Loan Default Predictor

Exploratory analysis of Bree's synthetic loan-applicant data, with a reference implementation of the existing rule-based policy. Intended as the data-generation / analysis stage prior to fitting models.

## Setup (Windows Git Bash or VS Code terminal)

```bash
python -m venv .venv
source .venv/Scripts/activate    # Git Bash on Windows
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate using `.venv\Scripts\Activate.ps1` instead.

## Reproduce the dataset and EDA

```bash
python data-analysis/run_generation.py
python data-analysis/analysis.py
```

- `data-analysis/generate_data.py` reproduces Bree's generation code **without modifications**. The wrapper runs it with the correct working directory, so the dataset lands at `data-analysis/loan_applications.csv` regardless of where you launch Python.
- `config.yaml` is the authoritative project configuration. Paths resolve relative to its location, not the current terminal directory.
- CSV reports and PNG plots appear in `data-analysis/outputs/`. They are small enough to commit so a reviewer can inspect the results without running the scripts.
- Commit the generated `data-analysis/loan_applications.csv` as required by the brief. Do **not** regenerate it with changed parameters.

## What the analysis checks

1. Data integrity, missingness and class counts.
2. Numeric feature dependencies (Spearman correlation heatmap), descriptive risk-signal associations, and observed training-set default rates across loan-to-income bins.
3. Reconstruction and decomposition of Bree's income verification, income level, bank stability, employment and cashflow rule factors.
4. Baseline default detection under two explicit three-to-two-action mappings (strict denial vs any intervention).
5. Employment-group approval/default rates on training data.
6. One-feature-at-a-time baseline **policy sensitivity**, reporting score changes and approval/denial flips.

All analyses involving `actual_outcome` use the **training split only**, leaving validation/test for model development and final evaluation. Splits are reproducible via `splits.make_splits()` and `config.yaml`.

## Interpretation cautions

- The CSV contains **rounded** deposits and withdrawals, whereas the provided rule score was computed from unrounded values. The baseline reconstructed from visible CSV inputs may differ slightly; `rule_based_score` and `rule_based_decision` are authoritative when scoring historical policy performance. See `outputs/baseline_reconstruction_check.json`.
- `ongoing` loans are **unlabeled**, not repaid. They are excluded from supervised training and primary outcome-conditioned analysis. Production data may suffer from non-random follow-up and label censoring.
- Policy contributions and one-variable stress tests describe **what the rule does**, not what predicts actual default and not causal effects. The synthetic generator itself exposes true generating rules; do not use hidden generator variables as predictive features.
- The rule-based model has three actions, not two. 'Flag for review' is reported explicitly and mapped transparently rather than silently treated as denial.
- Model construction, honest held-out testing and fairness analysis of **new model** decisions belong to the next phase.

For the analysis questions, assumptions and report mapping, see [`data-analysis/ANALYSIS_PROTOCOL.md`](data-analysis/ANALYSIS_PROTOCOL.md).
