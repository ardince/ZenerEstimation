# Milestone 20 — M20-A and M20-C1 Acceptance Record

**Project:** ZenerEstimation
**Record date:** 2026-11-26
**Scope:** M20-A architecture/source-interface inspection; M20-C1 authorized Grid Search real-model integration
**Decision:** Accepted at the tested software-contract level; Milestone 20 remains open.

## 1. Scope and acceptance decision

### M20-A — Architecture and source-interface inspection

**Decision:** Accepted for the reviewed architecture and interfaces.

Reviewed interfaces and contracts include:
- `FiniteSearchSpace` and its family-specific ARIMA and Kalman candidate mappings.
- Real `ARIMAForecaster` and `KalmanForecaster` constructor, `fit(dataset)`, and `predict(steps)` interfaces.
- The authorized execution pattern using `LocalOptimizationCoordinator`, `OptimizationExecutionRequest`, `ApprovedExecutionPlan`, and `AuthorizedOptimizationRunner`.
- Development/holdout isolation, validation protocol binding, and strict forecast timestamp alignment.

ARIMA candidates are passed as `{'order': (p, d, q)}`; Kalman candidates use `process_noise`, `drift_noise`, `regime_factor`, and `regime_multiplier`. The reviewed constructors accept these mappings without a parameter-renaming adapter.

### M20-C1 — Authorized Grid Search integration

**Decision:** Accepted for the tested ARIMA/Kalman integration and failure handling.

| Checkpoint | Coverage | Evidence |
| --- | --- | --- |
| C1-A | Real forecaster interface inspection | Source review completed |
| C1-B | Real ARIMA authorized Grid Search | 2 tests passed |
| C1-C | Real Kalman authorized Grid Search | 4 tests passed |
| C1-D | Real Kalman fitting-failure adversarial review | 2 tests passed |
| Cross-contract regression | C1-B/C/D, M20-B5-E, legacy ARIMA integration | 19 passed, 53 warnings |

Verified behaviors covered by these tests include bounded candidate execution, expanding-window folds, real model fitting, forecast date alignment, arithmetic mean of fold RMSE for ranking, best-candidate selection, authorization reuse rejection, and protected-holdout independence. The adversarial Kalman tests cover unsuccessful fit accounting without fabricated scores.

## 2. Test evidence and chronology

1. **Repository-wide baseline:** `python -m pytest -q` reported **3,813 passed, 161 warnings** after correcting the session-level OpenSSL configuration. This run preceded the addition/acceptance of the new M20-C1 tests; it must not be represented as a post-C1 full-suite result.
2. **C1-B focused:** `python -m pytest -q -v tests/test_m20_c1_b_arima_grid_real_path.py` reported **2 passed** after correcting the test fixture's development cutoff to `2024-12-01`.
3. **C1-B combined:** C1-B, M20-B5-E, and legacy ARIMA integration reported **13 passed, 53 warnings**.
4. **C1-C focused:** `python -m pytest -q -v tests/test_m20_c1_c_kalman_grid_real_path.py` reported **4 passed**.
5. **C1-C combined:** C1-B/C, M20-B5-E, and legacy ARIMA integration reported **17 passed, 53 warnings**.
6. **C1-D focused:** `python -m pytest -q -v tests/test_m20_c1_d_kalman_failure_review.py` reported **2 passed**.
7. **C1 consolidated:** The following command reported **19 passed, 53 warnings**:

```powershell
python -m pytest -q `
    tests/test_m20_c1_d_kalman_failure_review.py `
    tests/test_m20_c1_c_kalman_grid_real_path.py `
    tests/test_m20_c1_b_arima_grid_real_path.py `
    tests/test_m20_b5_e_real_path.py `
    tests/test_arima_optimization_integration.py
```

## 3. Known limitations and non-blocking observations

- **Kalman training history:** The current implementation initializes drift using `y[5]`; fewer than six training observations cause a fitting failure. The C1-D review exercises this real-model failure through the authorized runner. This record does not claim the model now validates minimum history with a custom exception.
- **ARIMA numerical warnings:** The consolidated regression reported 38 warnings concerning non-invertible initial MA parameters and 15 concerning non-stationary initial AR parameters, originating in the legacy ARIMA integration tests. They are non-blocking for this acceptance but remain diagnostics.
- **OpenSSL configuration:** An earlier full-suite attempt encountered 12 RFC3161 fixture setup errors because the active OpenSSL executable could not find `openssl.cnf`. Setting `OPENSSL_CONF` to an existing base-Anaconda configuration resolved certificate generation; the affected 12 tests subsequently passed, and the clean baseline reported 3,813 passes. The setting was session-scoped.
- **Scientific boundary:** Software-contract acceptance does not certify future forecast accuracy, superiority of either model, or performance on unseen battery measurements.
- **Regression boundary:** No clean repository-wide run incorporating the new C1-B/C/D tests has been reported yet. The 19-test consolidated run is the latest direct C1 acceptance evidence.

## 4. Change-control and Git checkpoint

This record documents test results supplied from the Windows ZenerEstimation repository. It does **not** assert that repository documentation has been edited, that a Git commit has been created, or that GitHub has been synchronized.

Proposed reviewed files for the checkpoint:
- `tests/test_m20_c1_b_arima_grid_real_path.py`
- `tests/test_m20_c1_c_kalman_grid_real_path.py`
- `tests/test_m20_c1_d_kalman_failure_review.py`
- `docs/milestones/M20_A_C1_Acceptance_Record.md` (this record, once copied into the repository)

Any updates to architecture, development-history, milestone-index, or release-note documents should be made only after checking their existing structure and conventions. Stage only reviewed paths; avoid blanket staging.

## 5. Remaining Milestone 20 work

- **M20-C2:** Authorized Random Search integration.
- **M20-C3:** Authorized Bayesian Search integration.
- **M20-C4:** Neural and hybrid-model integration.
- **M20-D:** Reproducibility and trial diagnostics.
- **M20-E:** Milestone-wide regression, evidence review, and final acceptance.

**Final disposition:** M20-A and M20-C1 accepted at the tested software-contract level. **Milestone 20 is not yet complete.**
