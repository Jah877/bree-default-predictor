# Bree — Loan Default Predictor

A reproducible comparison of Bree's rule-based loan screening policy with interpretable machine-learning models on the **synthetic** dataset provided in the Machine Learning Intern take-home. The repository includes the original dataset generator, generated CSV, exploratory analysis, four ML experiments, fairness checks, feature explanations, and a held-out evaluation.

**Finding:** Engineered logistic regression (E2) was selected on validation, but the final test results **do not demonstrate an unambiguous improvement** over the original rule-based policy. This is a research exercise, not a production-ready lending or denial system.

## Run the submission

Requirements: Python 3.10+; Python 3.13.5 was used for the verified run. From the repository root:

```bash
python -m venv .venv
```

Activate the environment with the command for your terminal:

```bash
# Windows Git Bash
source .venv/Scripts/activate

# OR Windows PowerShell
.venv\Scripts\Activate.ps1

# OR macOS / Linux
source .venv/bin/activate
```

Then install dependencies and run:

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
```

`run_pipeline.py` performs the data-quality analysis, all **11 automated tests**, and the E0–E4 validation comparison. It uses the committed CSV unless absent and **preserves the already-finalized test files**. To independently reproduce the completed held-out evaluation with the locked settings, run:

```bash
python run_pipeline.py --reproduce-final
```

For an initial run of this code without committed test outputs, `python run_pipeline.py --finalize-test` evaluates the test set once. `--regenerate` explicitly runs Bree's **unchanged** generator to recreate the CSV; the CSV is already included as required. Scripts use paths relative to `config.yaml` and do not depend on the terminal's working directory.

## Results — final held-out test

The 2,000-row generator yielded **1,291 repaid, 545 defaulted and 164 ongoing** applications. Bree's descriptive brief suggests approximately 78% repaid and 14% defaulted; the **actual executable generator** yields **64.55% and 27.25%**, respectively, plus 8.20% ongoing. We retained the generated data unchanged. Ongoing loans have unresolved labels and were excluded from the supervised comparison. The remaining 1,836 loans were stratified into 1,101 train, 367 validation and **368 test** rows (109 test defaults).

The primary operational definition is **intervention = review or denial**, not outright denial. A positive prediction means the application needs intervention. E0's stored decisions are authoritative; its saved scoring inputs are rounded, so numerical reconstruction differs slightly for 21 of 2,000 scores, although all 2,000 decisions are reproduced.

| System | Test TP | Test FP | Default recall | Default precision | Default F1 | FPR | ROC-AUC | Intervention rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **E0: Bree intervention baseline** | 78 | 99 | 71.6% | 44.1% | 0.545 | 38.2% | 0.724 | 48.1% |
| E1: Raw logistic regression | 83 | 108 | 76.1% | 43.5% | 0.553 | 41.7% | 0.714 | 51.9% |
| **E2: Engineered logistic regression (validation-selected)** | **80** | **108** | **73.4%** | **42.6%** | **0.539** | **41.7%** | **0.711** | **51.1%** |
| E3: Shallow decision tree | 93 | 120 | 85.3% | 43.7% | 0.578 | 46.3% | 0.700 | 57.9% |
| E4: E2 excluding employment | 78 | 110 | 71.6% | 41.5% | 0.525 | 42.5% | 0.718 | 51.1% |

These thresholds were frozen using **validation** predictions; they were **not retuned on test**. E3's coarse risk-score ties prevent matching the reference intervention volume as closely as the logistic models. The separate E0 *denial-only* analysis appears in [`test_metrics.csv`](modelling-experiments/outputs/test_metrics.csv), but it is not interchangeable with the review-or-denial primary comparison.

### Submission summary

**1. Data and handling.** Around 15% of documented income values are absent; some stated and documented incomes disagree substantially. Inputs include strongly correlated income, documented-income, and bank-deposit variables. On training data, missing documentation, higher loan burden, and overdrafts exhibit elevated observed default rates; these are **associations, not causes**. We inspected missingness, Spearman correlations, loan-burden default rates and the baseline's sensitivity to individual input changes. Numerical missing values are median-imputed using the training partition only. Missing documentation is additionally encoded as a flag in engineered models; categorical employment status is one-hot encoded. The `ongoing` class is excluded, rather than mislabelled as repaid. Three useful figures: [feature dependencies](data-analysis/outputs/figures/feature_dependencies.png), [loan-burden default rates](data-analysis/outputs/figures/loan_burden_default_rate.png), and [baseline employment-group behaviour](data-analysis/outputs/figures/employment_fairness.png).

**2. Models and features.** E0 measures Bree's existing rules. E1 fits L2 logistic regression to application-time attributes, and E2 adds four interpretable features: loan/income, income discrepancy/income, withdrawals/deposits, and an indicator for missing documentation. E3 tests nonlinearity using a shallow tree (depth four, minimum leaf size 30). E4 removes employment from the validation-selected model without changing its estimator family or other hyperparameters. E1 and E2 share identical estimator settings, making their comparison a controlled feature-representation test. E2 improved validation ROC-AUC from **0.698 to 0.730** versus E1 and was selected by a predeclared, intervention-volume-constrained default-recall criterion. On **validation**, permuting documented income or monthly deposits lowered E2 ROC-AUC most (by approximately 0.175 and 0.141, respectively); their interdependence makes isolated importance/causal claims inappropriate. Coefficient and applicant-level log-odds explanations are saved under [`importance/E2/`](modelling-experiments/outputs/importance/E2/) and [`example_decisions_validation.csv`](modelling-experiments/outputs/example_decisions_validation.csv).

**3. Results and operational tradeoff.** On the held-out test, E2 caught **two additional defaults** versus E0 (**80 vs 78 of 109**) but flagged **nine additional repaying applicants** (**108 vs 99 of 259**). Its false-positive rate was higher (41.7% vs 38.2%), F1 was slightly lower (0.539 vs 0.545), and ROC-AUC was lower (0.711 vs 0.724). Test intervention volume was 51.1% for E2 versus 48.1% for E0, despite matching on validation; this is a limitation of fixed-threshold generalization. The data therefore **do not support deployment as a clear replacement**. E3 caught more defaults but flagged many more applicants, so its higher recall is not evidence of a better like-for-like decision policy. Without lending-loss or review-cost estimates, no threshold can be declared economically optimal. See [test confusion matrices](modelling-experiments/outputs/figures/test_E2_confusion.png), [validation ROC](modelling-experiments/outputs/figures/validation_roc.png), and [test probability calibration](modelling-experiments/outputs/figures/test_calibration.png). E2's test Brier score is 0.183; no separate probability calibrator was fitted.

**4. Fairness.** On the held-out test, employed and self-employed applicants had observed default rates of **28.2%** and **31.1%**, respectively. Automatic approval rates were **65.7% vs 41.0%** under E0 (24.7-percentage-point gap), **59.2% vs 44.3%** under E2 (14.9 pp), and **54.9% vs 45.9%** under E4 (9.0 pp). E4 reduced the measured approval gap but caught only 78 defaults while flagging 110 repayers. E2 automatically approved none of the 33 unemployed test applicants; E4 approved seven, two of whom defaulted. These subgroup counts are small. Removing employment is a useful diagnostic, **not proof of equitable treatment**: income and deposit features can encode related information. Approval rates, group-level recall and FPR all require continued monitoring. See [employment fairness by experiment](modelling-experiments/outputs/test_employment_fairness.csv).

**5. Production risk and next investigation.** The most immediate risk is the **synthetic-to-real-data gap**. In the generator, repayment outcomes exist even for applicants Bree would deny; in actual lending, repayment labels generally are unavailable for denied applicants (selective-label bias). Actual ongoing loans may also be nonrandomly censored. Recession-driven shifts, changing applicant mix, missing documentation, drift in risk calibration, subgroup disparities and review capacity would require validation before any real-world use. The ML outputs currently describe **binary intervention**, not a legally and operationally complete approve/review/deny workflow. No real credit decisions should be made from this prototype. A responsible next step would be a governed prospective shadow evaluation with human oversight, explicit intervention costs, calibration, adverse-action explanations and subgroup monitoring.

## Repository guide

```text
config.yaml                       Shared paths, split and decision settings
data-analysis/
  generate_data.py               Bree's original generator (unchanged)
  run_generation.py              Runs generator from the correct directory
  loan_applications.csv          Committed synthetic dataset
  analysis.py, baseline.py       Exploratory and original-policy analyses
  splits.py                      Reusable stratified partition
  outputs/                       Training-only EDA artifacts
modelling-experiments/
  experiments/e0_*.py ... e4_*.py  Individually specified experiments
  features.py, specs.py          Leakage-aware transformations and pipelines
  evaluation.py, explain.py      Metrics, fairness and explanation functions
  run_experiments.py             Experiment comparison and final evaluation
  tests/                        Automated checks
  outputs/                       Validation and final test artifacts
run_pipeline.py                   Single reproducibility command
requirements.txt                 Python dependencies
```

The complete experiment assumptions, model-selection rule, and leakage controls are documented in [`METHODOLOGY.md`](modelling-experiments/METHODOLOGY.md). The EDA protocol is documented in [`ANALYSIS_PROTOCOL.md`](data-analysis/ANALYSIS_PROTOCOL.md). Outputs, plots, and the final selection manifest are included in the repository so the evaluation can be audited without rerunning models.
