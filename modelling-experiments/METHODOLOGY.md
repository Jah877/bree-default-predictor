# E0–E4 methodology, assumptions and experiment log

This document is intended to be defendable in a take-home interview. It
explains **why** each experiment exists; numerical conclusions are preliminary
until the final held-out test is evaluated **once**.

## 1. What the labels actually mean

The supplied generator implements **two distinct mechanisms**:

- `rule_based_score` and `rule_based_decision` are computed by a fixed,
  weighted, deterministic policy based on application features.
- A **separate**, synthetic latent risk equation computes the probability of
  default. A Bernoulli sampler draws default outcomes; the generator then
  randomly labels approximately 8% of applications `ongoing`.

The correct supervised task is to predict `defaulted` vs `repaid` from
**application-time inputs**, treating the latent generator formula as unknown.
Do **not** copy the formula or train using `days_to_default`, the rule score,
rule decision, applicant ID, or the label itself. There is irreducible outcome
randomness: even the true risk probability does not perfectly predict defaults.

**Synthetic-world caveat:** the generator assigns realized repayment labels
even to denied applicants. In real lending, denied borrowers lack observed loan
repayment outcomes (selective labels), so the data do not replicate the
production deployment problem. `ongoing` is randomly selected in this generator;
real-world unresolved loans can be censored in a nonrandom, time-dependent way.

## 2. Split discipline / leakage prevention

- Generate the original 2,000-row CSV *once* using the unmodified supplied code.
- Exclude `ongoing` from supervised classification (never treat as repaid).
- Reuse `data-analysis/splits.py`: **60% train / 20% validation / 20% test**,
  stratified on the observed binary label, using seed 42.
- Study associations and select features on training records. Fit **all**
  imputers, encoders, scalers, and models on training records only.
- Use validation outcomes to **compare E1/E2/E3**, to study explainability, and
  to inspect E4. Choose thresholds using validation *predicted probabilities*
  and the baseline intervention frequency, without accessing labels.
- Evaluate the fixed original pipelines and locked thresholds on the test set
  **once** (`--finalize-test`). Do not retune afterward.
- Candidate models are **not** retrained on train+validation before final
  testing: retraining would change their scores and invalidate the validation-
  selected thresholds without a new calibration step.

## 3. Models and controlled comparisons

| ID | Input representation | Estimator | Variable isolated |
| --- | --- | --- | --- |
| E0 | Stored Bree policy scores/decisions | Deterministic rules | Existing policy performance |
| E1 | Raw applicant-time features | L2-regularized logistic regression | Learned model vs rules |
| E2 | Raw + 4 engineered financial indicators | Same logistic settings as E1 | Value of transparent feature engineering |
| E3 | Same engineered features as E2 | Shallow decision tree (max depth 4, min leaf 30) | Value of nonlinear splits/interactions |
| E4 | Best validation-selected E1/E2/E3 inputs, excluding employment | Identical estimator and parameters as selected parent | Employment ablation |

Detailed, prespecified model parameters are in individual files under
`experiments/`; shared decision policy lives in `config.yaml`.

- **E1 and E2:** `C=1.0`, L2 penalty (scikit-learn default), `lbfgs`, 3,000
  iterations, no class weighting. Model probability scores can therefore be
  assessed with Brier loss; class imbalance is explicitly addressed with
  positive-class recall, FPR, PR-AUC, and threshold selection, rather than by
  silently modifying the observed outcome prevalence.
- **E3:** no tuning grid or test-based hyperparameter search; a deliberately
  small tree controls variance and retains inspectable decision paths.
- **Engineered features:** loan amount / stated income; absolute discrepancy
  between documented and stated income relative to stated income; withdrawals
  / deposits; explicit documentation-missing flag. When a denominator is zero
  or a value is missing, keep NaN and impute inside the training-fitted pipeline.
  These are derived from observable inputs, not hard-coded copies of the hidden
  generator thresholds.
- **E4:** remove *only* employment status. Other correlated features may act as
  employment proxies. Removing employment does not establish fairness.

## 4. A pre-specified operational comparator

Bree's policy has **three actions**, whereas the learned classifier emits a
binary indicator for **intervention** (not an actual denial decision).

- **Primary E0 mapping:** `approved` = automatic approval; `flagged_for_review`
  OR `denied` = intervention. We are comparing *intervention rates*, not denial
  rates. This is a deliberately simplified operational mapping.
- **Secondary E0 mapping:** only `denied` is treated as predicted default.
  This must be separately labeled; it flags far fewer people.
- **Risk ranking:** `1 - rule_based_score/100` for E0 ROC-AUC and PR-AUC. This
  is **not a calibrated probability**; E0 has no Brier score.

For each learned model, choose an independent validation threshold that matches
E0's **validation intervention fraction** as closely as possible. For trees,
identical predicted probabilities may make an exact match impossible. We report
actual intervention rates instead of hiding this discrepancy.

**Parent selection rule, specified before test:** E1–E3 are eligible only when
their intervention rate differs from the reference rate by at most **2
percentage points**. Among eligible candidates, choose highest **default
recall**, then **average precision**, then **ROC-AUC**, then simpler experiment
in order E1 → E2 → E3. E4 is a diagnostic ablation of the winning parent and
is *not automatically promoted to winner* based on its held-out test result.

Without explicit business costs, the resulting intervention-rate-matched
comparison is a useful operational proxy, *not* proof of economic optimality.
A financial decision-maker would need default losses, intervention costs,
credit access impacts, and regulatory review before deployment.

## 5. What each metric and output means

`outputs/validation_metrics.csv` and (only after finalization)
`outputs/test_metrics.csv` include:

- TP: defaults flagged; FN: defaults missed;
- FP: repaying applicants flagged; TN: repaying applicants approved;
- default precision, recall, F1, FPR, FNR and intervention rate;
- ROC-AUC, average precision (PR-AUC summary), and Brier loss (learned
  probability models only);
- `additional_defaults_caught_vs_E0` and
  `additional_good_applicants_flagged_vs_E0` are **absolute counts on the same
  individuals** under the primary E0 intervention mapping.

`outputs/validation_employment_fairness.csv` reports per-employment-group
approval rates, observed default rates, recall, FPR and default rates among
approved applicants, with group sample sizes. This is descriptive evidence,
not a definitive finding of discrimination; evaluate sample uncertainty and
possible correlations with other financial factors.

`outputs/importance/E*/` provides standardized logistic coefficients or
impurity importance for the tree, plus **permutation importance**, measured as
the decrease in validation ROC-AUC when an original input is shuffled. Because
ratios share underlying inputs, correlated variables can mask or redistribute
importance. Coefficients describe log-odds, not causal effects. Permutation
importance is measured on the unchanged trained pipeline; it does not retrain.

`outputs/example_decisions_validation.csv` provides actual reviewer examples:
logistic contributions to **log-odds** or the precise decision-tree splits.
A model output is an indication of estimated risk, *not* an explanation of what
caused a historical loan to default.

`outputs/selection_manifest.json` records exact parameters, winning E1–E3
parent, E4 parent, thresholds, intervention definition and whether the test was
unlocked. `outputs/validation_predictions.csv` permits manual audit of the
results. Figures include ROC, precision–recall and confusion-matrix plots.

## 6. Preliminary experiment log (validation only)

After the first validation run (n=367, defaults=109):

| Experiment | Defaults caught (TP) | Repaid applicants flagged (FP) | Intervention rate | Interpretation |
| --- | ---: | ---: | ---: | --- |
| E0 intervention | 77 | 109 | 50.68% | Reference |
| E1 raw logistic | 77 | 109 | 50.68% | No count improvement at fixed volume |
| E2 engineered logistic | 82 | 104 | 50.68% | +5 defaults caught, 5 fewer good applicants flagged |
| E3 shallow tree | 90 | 130 | 59.95% | Higher recall, **but not comparable at matched volume** |
| E4 E2 without employment | 78 | 108 | 50.68% | Fairness/performance diagnostic |

E2 is the selected parent **under the pre-specified validation rule**. These
figures are not final test performance. The tree might improve with probability
calibration or different settings, but such an extension would be a separately
pre-registered experiment rather than test-driven tweaking.

## 7. Open questions before any deployment recommendation

- How sensitive are operating points to real economic cost ratios and applicant
  review capacity? `flagged_for_review` is not the same as rejected credit.
- Are probability estimates calibrated across risk levels and employment
  groups? Overall Brier loss alone does not guarantee group calibration.
- How uncertain are subgroup metrics for the small unemployed sample?
- What happens to approvals and performance under covariate/concept drift?
- What bias is introduced by conditioning on observed loan outcomes in a real
  institution where denied applicants have no repayment data?
- How could legal compliance, notice/reason requirements, audit trails,
  monitoring, and human review be built into an actual credit-decision process?

## 8. Reproducing results

From repository root after installing `requirements.txt`:

```bash
python data-analysis/run_generation.py
python data-analysis/analysis.py
python -m unittest discover -s modelling-experiments/tests -v
python modelling-experiments/run_experiments.py
# ONLY after validation design is frozen:
python modelling-experiments/run_experiments.py --finalize-test
```

Final test files are never generated by default. If an earlier test report is
present, the runner blocks re-finalizing unless `--overwrite-test` is given
(for deliberate exact reproducibility reruns only, never model selection).
