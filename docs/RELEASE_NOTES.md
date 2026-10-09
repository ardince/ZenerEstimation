# ZenerEstimation Release Notes

**Document** : RELEASE_NOTES.md  
**Framework Version** : 0.7.0  
**Document Version** : 0.7.0  
**Status** : Active  
**Last Updated** : July 2026

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