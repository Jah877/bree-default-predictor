# Experiment methodology and audit trail

## Question and data-generating process

The modelling task is to predict an observed synthetic `defaulted` versus `repaid` outcome from application-time features, not to imitate Bree's stored score. Bree's rule-based decision is a deterministic weighted score, while the take-home's **separate synthetic probability mechanism** creates the repayment outcomes. The generator's latent risk variables and probability function were not passed to any model. Outcome randomness means the observable inputs cannot perfectly predict default. The dataset is synthetic and is not representative evidence of an actual lending portfolio.

The original generator was preserved, and its output CSV was committed. The realized label mix (1,291 repaid / 545 defaulted / 164 ongoing) differs from the approximate mix in the narrative. Ongoing applications have unknown outcomes and are excluded from supervised training/evaluation. The known-outcome dataset is split stratified and reproducibly with seed 42 into 1,101 training, 367 validation and 368 test records.

## Feature availability and leakage controls

The ML input allowlist excludes `applicant_id` (except for reporting identity), `actual_outcome`, `days_to_default`, `rule_based_score` and `rule_based_decision`. Ratios and missingness flags are deterministically constructed from remaining application-time information. Logistic regression uses training-fitted median imputation, categorical one-hot encoding and numeric standardization; the tree omits scaling. All learned transformations are encapsulated in a scikit-learn pipeline and fit **only on training data**. Validation determines model selection and policy threshold. The held-out test is evaluated once with the fixed models and thresholds, without retraining or retuning.

## Predefined experiments

| Experiment | Inputs / estimator | Hypothesis | Parameters |
| --- | --- | --- | --- |
| E0 | Original rule score and stored decisions | Fixed operational comparator | Automatic approve >=75, review 50–74, deny below 50 |
| E1 | Original application fields + logistic regression | Data-fitted weights can improve fixed rules | L2, `C=1.0`, `lbfgs`, `max_iter=3000`, no class weights |
| E2 | E1 plus four engineered indicators | Affordability and cash-flow ratios add signal | Same estimator and parameters as E1 |
| E3 | Engineered inputs + decision tree | Nonlinear splits/interactions add signal | `max_depth=4`, `min_samples_leaf=30`, `min_samples_split=60`, Gini |
| E4 | Selected E1–E3 model without employment | Test group-disparity/performance sensitivity | Same parent estimator and hyperparameters |

E2 adds `loan_amount/stated_monthly_income`, `abs(stated_monthly_income - documented_monthly_income)/stated_monthly_income`, `monthly_withdrawals/monthly_deposits`, and `documentation_missing`. Undefined ratios remain NaN and are imputed using training statistics, retaining the documentation-missing indicator. E4 is a diagnostic ablation, not an alternative model promoted using test results.

## Operational comparator and selection rule

A positive prediction denotes **intervention**: human review or denial. The primary E0 baseline treats `flagged_for_review` and `denied` as interventions; a separate E0 strict denial-only metric is published for transparency. These are different operating policies. E0's negative rule score ranks default risk but is not a calibrated probability and has no Brier score.

Model probabilities are thresholded at values chosen using **validation predictions only** to approximate E0's validation intervention rate (186/367 = 50.68%). This threshold calculation does not inspect validation outcome labels. Tree probabilities have large tied groups, so a shallow tree cannot always reproduce the exact volume; observed intervention rates are always reported. An E1–E3 model is eligible for selection if its validation intervention rate is within **two percentage points** of E0. Eligible models are ranked by validation default recall, then average precision, then ROC-AUC, then E1/E2/E3 simplicity order. E2 won under this fixed criterion. It was never replaced after observing test results.

For every policy, confusion counts use defaulted = positive, intervention = predicted positive; FP counts repayers flagged and FN counts defaults automatically approved. The same definitions apply to precision, recall, F1, FPR and FNR. ROC-AUC and average precision evaluate continuous risk ranking. Brier scores and six-quantile calibration curves diagnose the probability quality of the learning models; **no probability calibrator is fitted**. Prediction and fairness tables include each group's sample size to prevent overinterpreting small subgroup rates.

## Locked validation and final test

| Experiment | Validation TP/FP | Validation intervention | Final test TP/FP | Final test intervention | Final test ROC-AUC |
| --- | --- | ---: | --- | ---: | ---: |
| E0 review-or-deny | 77 / 109 | 50.7% | 78 / 99 | 48.1% | 0.724 |
| E1 raw logistic | 77 / 109 | 50.7% | 83 / 108 | 51.9% | 0.714 |
| **E2 engineered logistic** | **82 / 104** | **50.7%** | **80 / 108** | **51.1%** | **0.711** |
| E3 shallow tree | 90 / 130 | 59.9% | 93 / 120 | 57.9% | 0.700 |
| E4 E2 without employment | 78 / 108 | 50.7% | 78 / 110 | 51.1% | 0.718 |

The E2 test comparison to E0 is **+2 defaults detected and +9 repayers intervened upon**. E2 test default F1 (0.539) and AUC (0.711) were below E0 (0.545 and 0.724). Thus a validation improvement did not produce an unqualified test improvement. This is a limited sample, not proof that the learned model generalizes better or worse across all future populations. E3 detects more defaults while intervening on substantially more applicants; the comparison is not capacity-matched.

## Explanations, fairness and limitations

Global permutation importance measures the change in **validation** ROC-AUC when an original applicant field is shuffled; derived features are recalculated consistently with the shuffled input. Correlated inputs can make attribution ambiguous. Logistic coefficients and per-person log-odds contributions show the learned prediction mechanism, **not a causal explanation** for a historical default.

Group reports stratify by employment status with observed default rates, approval/intervention rates, default recall, FPR and default rates among approved applicants. In final test, employed/self-employed approval gaps decreased from 24.7 percentage points for E0 to 14.9 points for E2 and 9.0 points for E4. This specific parity measure improved, but subgroup error rates differ and E2 approved none of the 33 unemployed applicants. E4 approved seven unemployed applicants, two of whom defaulted. The dataset is small and synthetic; these outcomes cannot establish the fairness, lawfulness or safety of a real credit policy. Removing employment also does not remove proxies.

**Deployment limitations:** The synthetic generator observes repayment for even denied applicants, unlike real-world selective lending labels; `ongoing` cases can be censored in production, and review outcomes themselves may alter observed risk. Deployment would require real applicant data, human and legal oversight, a well-defined three-way decision policy, explicit relative costs of errors, calibration, drift monitoring, reliable reviewer-facing reasons, and fairness monitoring. None of these is validated here. This package is reproducible for an internship assessment, **not production-ready credit decision software**.

## Reproduction

From the repository root, install `requirements.txt`, then use `python run_pipeline.py` for dataset analysis, automated tests and validation. Use `python run_pipeline.py --reproduce-final` to recompute the already locked held-out test evaluation. The stored dataset SHA-256, parent selection, fixed hyperparameters and thresholds are in [`outputs/selection_manifest.json`](outputs/selection_manifest.json). The runner checks those fields before rerunning a finalized experiment and refuses silent drift.
