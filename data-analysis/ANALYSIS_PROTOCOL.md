# Data-analysis protocol — before model selection

## Objective

Understand the synthetic data, reconstruct Bree's current policy, and separate (a) factors **explicitly rewarded or penalized by the policy** from (b) characteristics **observationally associated with default**. Preserve an untouched test partition for an honest model comparison.

## 1. Reproducibility and integrity

- Execute `generate_data.py` **unchanged** through `run_generation.py`; commit the output CSV.
- Check row count, unique IDs, field types, nulls and outcome / policy-action distributions.
- Compare actual label proportions with Bree's textual description, and describe discrepancies rather than manipulating data to meet expectations.
- Keep `ongoing` records for descriptive counts, but do not treat them as repaid or defaulted in supervised binary training.

**Outputs:** `data_quality.csv`, `outcome_counts.csv`, `baseline_decision_counts.csv`.

## 2. Hold out data before outcome-related analysis

- Use seeded stratified 60/20/20 train/validation/test on **repaid/defaulted cases only** (`splits.py`).
- Explore associations, default rates, rule metrics and fairness outcomes on **training** records only. Inspect validation/test performance tomorrow when fitting and selecting models.
- Be wary that missing outcomes can create censoring and selection bias in real lending portfolios (the synthetic generator randomly marks ongoing).

## 3. Feature quality, dependency, and descriptive signals

- Examine missing documented income, discrepancies between stated and documented income, loan-to-income, withdrawal/deposit ratio, bank overdrafts and balance.
- Use Spearman correlations to identify redundant numeric variables; correlations alone don't measure prediction value.
- Compare training-set conditional default rates for understandable financial indicators and across loan-burden quantile bins; include sample sizes to show uncertainty.
- Distinguish synthetic data relationships from plausible real-world financial intuition.

**Outputs:** `spearman_feature_correlations.csv`, `descriptive_risk_signals_train.csv`, `loan_burden_default_rate_train.csv`, and the associated plots.

## 4. Reverse engineer and evaluate Bree's baseline

- Compute five raw sub-scores and weighted contributions (income verification 30%, income level 25%, account stability 20%, employment 15%, withdrawals/deposits 10%).
- Rebuild policy actions from a score threshold: approve >=75, review >=50, otherwise deny.
- Compare reconstructed scores/decisions to *stored* baseline results; use stored data as the ground truth for historical baseline metrics because some input values were rounded before CSV export.
- For a binary comparison with actual outcomes, report two explicit mappings: **denied-only** predicts default vs **denied-or-review** predicts default. Compute default precision/recall/F1, confusion counts, false-positive/negative rates and ROC-AUC using negative rule score (lower score means more default risk).

**Outputs:** `baseline_component_summary_train.csv`, `baseline_reconstruction_check.json`, and `baseline_metrics_train_only.csv`.

## 5. Sensitivity and fairness

- Change *one observed input at a time*, recalculate baseline component scores and report policy decision flips. The sensitivity measures **policy responsiveness**, not causes of default, not independently realistic changes to a person's finances, and not feature importance for a future predictive model.
- Employment-status what-ifs are restricted to the relevant source groups (employed to self-employed and vice versa), making Bree's explicit employment penalty inspectable.
- Compare baseline approval, review, denial and observed group default rates by employment status. In the next phase add new-model group rates and group-specific error rates.

**Outputs:** `baseline_sensitivity_train.csv`, `employment_fairness_train.csv` and `employment_fairness.png`.

## Modelling decision tomorrow

Start with a transparent, carefully preprocessed logistic regression; test whether a shallow decision tree's nonlinear splits improve held-out risk detection. Do not tune using test outcomes. Compare with the same baseline sample and explicit decision threshold/mapping. Examine calibration, group-specific error rates and case-level explanations before considering an ensemble or deployment recommendation.
