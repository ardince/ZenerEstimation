# ZenerEstimation Release Notes

**Document** : RELEASE_NOTES.md  
**Framework Version** : 0.7.0  
**Document Version** : 0.7.0  
**Status** : Active  
**Last Updated** : 18 November 2026

---

# Unreleased — Sprint 14.9 Transition Diagnostics

This is a development-status entry, **not** a new numbered release. The existing v0.7.0 release history below is unchanged.

## Added

- Dataset-410 benchmark-history and benchmark-to-operational transition visualizations.
- Reusable evidence objects for benchmark and operational runner stages.
- Regression coverage for transition plot boundaries, model matching, and execution reuse.

## Clarified

- Frozen benchmark: training through 2023-12-01; six measured holdout quarters from 2024-03-01 to 2025-06-01.
- Operational extrapolation: refitting through 2025-06-01 using unchanged configurations; six **unvalidated** future quarters from 2025-09-01 to 2026-12-01.
- Diagnostic plotting does not introduce additional model fitting or prediction.
- Neural model optimization remains deferred; no benchmark results were modified during the source review.

## Verification

- User confirmed successful execution of the complete dataset-410 runner and full test suite.
- No new numerical benchmark metrics or release version are asserted in this entry.

## Unreleased — Sprint 14.9G (dataset 110)

### Added
- Frozen five-quarter measured-holdout benchmark for battery `732B-5610110`, using transferred battery-410 configurations.
- Independent six-quarter operational extrapolation through 2026-09-01, explicitly labeled unvalidated.
- Two-panel transition diagnostics generated exclusively from stored evidence.
- Preflight, execution, operational and transition regression tests.
- Human-readable two-decimal forecast CSV with separate full-precision scientific evidence.

### Fixed
- `ForecastEvaluator.split_dataset()` now splits the historical endpoint-filtered data when `evaluation_end` is specified, preventing later observations from entering the historical benchmark.

### Scientific safeguards
- Holdout measurements are never imputed or used for hyperparameter/seed selection.
- Operational predictions are not benchmark accuracy evidence.
- Transition plotting does not fit models or generate predictions.
- Frozen benchmark metrics and rankings are unchanged by display formatting.

### Release status
Pending final local regression suite, Git diff review, and user-approved commit/push.


---

# Unreleased — Milestone 16 (local development checkpoint)

This entry documents completed implementation and local validation, **not** a new numbered release or a GitHub publication. Framework version metadata remains at **0.7.0**.

## Added

- Explicit quarterly forecast-alignment evidence and optional strict enforcement in optimization evaluation.
- LSTM/GRU training diagnostics, neural optimization evidence, and provenance-linked reporting.
- Dataset and training-partition fingerprints, environment provenance, and experiment manifests.
- Cross-battery historical-fit, external-accuracy, optimized-neural, and numerical-comparison views, including dashboard and Excel exports.
- Numerical reconciliation checks for saved results and measured-value enrichment.

## Verification

- Final local regression suite: **1,891 passed, 161 warnings, zero failures** (November 2026).
- The warnings were non-failing; this entry does not imply that all underlying numerical-library warnings have been resolved.

## Compatibility and limitations

- Strict alignment enforcement remains **opt-in** (`require_verified_alignment=False` by default).
- Historical fits, measured holdout evaluations, and unvalidated operational projections retain distinct interpretations; future operational estimates are not reported as measured accuracy.
- Minor dashboard presentation refinements and additional accuracy-optimization UI work are deferred to Milestone 17.
- Local Git commit and GitHub synchronization were **pending** when this note was prepared.

---

# Unreleased — Milestone 19 D2 (internal ranking sensitivity; 24 November 2026)

## Added

- M18-verified internal fold RMSE extraction and retrospective leave-one-fold-out score influence.
- Common-fold cross-model ranking sensitivity with exact-tie competition ranks and winner-set changes.
- Aggregate internal rank/winner-set stability fractions, change counts, maximum rank displacement, and per-model change counts.
- End-to-end D2 cross-contract tests covering evidence consistency and fail-closed behavior.

## Verification

- D2-D focused integration: **18 passed**.
- Selected M18/D2 regression: **170 passed**.
- Full repository: **3,560 passed, 161 warnings**, exit code 0, 1664.57 seconds (2026-11-24).

## Limitations and release status

These are read-only retrospective internal diagnostics; LOFO does not retrain models. Stability fractions are not prospective performance probabilities. D2 does not authorize protected-holdout access, independent scientific certification, or model selection. The 161 reported warnings remain unresolved. This is an **unreleased local software-contract acceptance**, not a numbered release or GitHub publication.


## Unreleased — Milestone 19 D1 (software-contract acceptance, 23 November 2026)

This is a development acceptance entry, not a numbered release or GitHub publication.

### Added and validated

- Verified M18 selection / M17 internal fold evidence, preserving ranking versus pooled RMSE.
- Fail-closed protected-holdout boundary and independent M19-C2 decision reconciliation.
- D1-C1 read-only generalization-gap assessment facade, reporting no numerical gap or comparability.
- Downstream B1–B4 separation and 14 adversarial test cases.

### Verification

- D1 combined regression: **86 passed** across eight test files.
- Full repository regression: **3,481 passed, 161 warnings in 1,616.84 seconds (26:56), exit code 0**; previous baseline 3,445 passed, 161 warnings.

### Limitations

- No independent scientific authorization, external authentication, prospective access permission, benchmark participant binding, protected-holdout comparison/ranking, or numerical generalization gap is established.
- D1-C2 numerical assessment remains deferred; acceptance is limited to tested software contracts.
- Existing numerical/ARIMA and Keras warnings remain; this entry does not assert they were fixed.

**Record:** `docs/milestones/M19_D1_Final_Acceptance_Record.md`.

---

# Unreleased — Milestone 19 B2-I2 (software-contract acceptance)

**Accepted:** 18 November 2026. This is a local development acceptance entry, **not** a numbered release. Framework version remains **0.7.0**; no Git commit or GitHub publication is asserted.

## Added

- Temporal eligibility findings bound to M18 plan and execution evidence.
- Verified target-observability and coverage reporting with provenance-aware exclusions.
- Verified forecast-to-observation alignment and observed-only RMSE, MAE and MAPE.
- Integrated, traceable assessment evidence with identity, digest, calendar and metric consistency checks.
- Cross-contract adversarial tests for evidence manipulation, temporal-status precedence, scoring denominators and snapshot isolation.

## Verification

- Combined I2 regression: **136 passed** across nine test files, zero reported failures/errors.
- Separately executed targeted M18 plan/execution regression: **89 passed** across two test files, zero reported failures/errors.
- Total: **225 passing test cases across two runs**, not one 225-test invocation or a full repository regression.

## Limitations

- Acceptance is limited to software-contract validation; independent measurement authentication and scientific certification are **not granted**.
- `verified_findings` and available observed-only metrics must not be represented as proof of prospective scientific validity.
- Broader M18 and repository-wide regression are outside the reported targeted verification scope.

**Record:** `M19_B2_I2_Final_Acceptance_Record.md`.

---

# v0.7.0

## Added

- Generic prognostics framework
- ThresholdEstimator
- MonteCarloRUL
- RULAnalyzer
- PrognosticResult
- demo_rul.py

## Improved

- Kalman forecasting workflow
- Metadata generation
- Report generation
- Documentation

## Testing

- 62 unit tests passing

---

# v0.6.0

## Added

- Adaptive Kalman forecaster
- demo_kalman.py
- Bias correction
- Residual estimation
- Kalman metadata

---

# v0.5.x

## Added

- ARIMA forecaster
- Forecast reports
- Forecast plots
- Experiment registry
- Metadata export

---

# Earlier Versions

Earlier releases established

- Project structure
- Dataset infrastructure
- Generic forecasting interfaces
- Visualization framework

These releases formed the basis for subsequent forecasting
and prognostics development.

---

End of Document