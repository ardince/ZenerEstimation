# M20-C4 — Final Acceptance Record

**Project:** ZenerEstimation
**Milestone:** M20 — Authorized Optimization
**Workstream:** M20-C4 — Neural and Hybrid Model Integration
**Record date:** 2026-11-27
**Acceptance scope:** Software-level integration and contract verification
**Technical status:** ACCEPTED
**Documentation status:** Prepared for repository review and commit

## 1. Purpose and acceptance boundary

This record consolidates the evidence for M20-C4 integration of four forecasting families into the authorized optimization workflow: `LSTM`, `GRU`, `LinearTrendLSTM`, and `KalmanLSTM`.

Acceptance concerns the behavior exercised by the recorded tests: authorized execution, development-only temporal validation, forecast alignment, candidate ranking, epoch-fold budget accounting, reproducibility, and failure evidence. It does **not** certify accuracy on protected future observations, superiority of one model family over another, universal TensorFlow determinism, or production deployment readiness.

## 2. Scope and completed stages

| Stage | Scope | Recorded result |
| --- | --- | --- |
| M20-C4-A | Neural/hybrid source and optimization-contract inspection | Completed |
| M20-C4-B1 | Real-path authorized LSTM Grid Search integration | 4 passed |
| M20-C4-B2 | Real-path authorized GRU Grid Search integration | 4 passed |
| M20-C4-C1 | Real-path authorized LinearTrendLSTM Grid Search integration | 4 passed |
| M20-C4-C2 | Real-path authorized KalmanLSTM Grid Search integration | 4 passed |
| M20-C4-D1 | Combined four-family regression | 16 passed; 10 warnings; 130.12 s |
| M20-C4-D2 | Cross-model reproducibility | 4 passed; 4 warnings; 75.36 s |
| M20-C4-D3 | Budget and evidence consistency | 8 passed; 4 warnings; 70.23 s |
| M20-C4-D4 | Consolidated cross-model acceptance regression | **28 passed; 18 warnings; 177.79 s** |

The D4 total includes the 16 family-integration tests, four D2 tests, and eight D3 tests. These are 28 distinct tests in one consolidated run; the earlier stage runs are not additional unique test cases.

## 3. Consolidated verification command and result

Executed from the repository root in the Windows `zener` environment:

```powershell
python -m pytest `
    tests/test_m20_c4_b1_lstm_grid_real_path.py `
    tests/test_m20_c4_b2_gru_grid_real_path.py `
    tests/test_m20_c4_c1_linear_trend_lstm_grid_real_path.py `
    tests/test_m20_c4_c2_kalman_lstm_grid_real_path.py `
    tests/test_m20_c4_d2_cross_model_reproducibility.py `
    tests/test_m20_c4_d3_budget_evidence_consistency.py `
    -q --tb=short
```

Recorded terminal summary:

```text
............................ [100%]
28 passed, 18 warnings in 177.79s (0:02:57)
```

**Recorded environment:** Windows (`win32`); Python 3.11.15; pytest 9.1.1; Conda environment `zener`. The D2/D3 logs additionally report pluggy 1.6.0 and anyio 4.15.1.

## 4. Contract evidence and findings

### 4.1 Authorization, isolation, and temporal alignment

The accepted real-path integration tests exercise authorized optimization execution, fresh model construction within validation folds, fold-local training, forecast-date alignment, and protection of declared holdout observations. The evidence supports these contracts for the tested model families, datasets, and configurations; it is not a universal proof for all possible inputs.

### 4.2 Reproducibility (M20-C4-D2)

Each model family was exercised in paired independently authorized runs. The reported candidate ranking scores and maximum fold-RMSE differences were:

| Family | First score | Second score | Absolute score difference | Maximum fold-RMSE difference |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 12.4237816349 | 12.4237816349 | 0 | 0 |
| GRU | 12.2386403338 | 12.2386403338 | 0 | 0 |
| LinearTrendLSTM | 0.453938253155 | 0.453938253155 | 0 | 0 |
| KalmanLSTM | 0.422577365003 | 0.422577365003 | 0 | 0 |

All four D2 tests passed. These results establish structural and numerical reproducibility in the exercised environment and configurations. They do **not** establish bitwise equality of all weights, cross-platform determinism, or reproducibility across arbitrary TensorFlow versions or hardware.

### 4.3 Successful-trial budget and ranking evidence (M20-C4-D3)

For every family, the D3 successful-execution test reported two candidates evaluated over two validation folds:

- Candidate 1: **4 reserved epoch-folds**; observed training epochs `[2, 2]`.
- Candidate 2: **6 reserved epoch-folds**; observed training epochs `[3, 3]`.
- Total per-family reservation in the exercised successful case: **10 epoch-folds**.

Reported ranking RMSE values:

| Family | Candidate 1 | Candidate 2 |
| --- | ---: | ---: |
| LSTM | 12.4237816349 | 12.0153076256 |
| GRU | 12.2386403338 | 11.7384587512 |
| LinearTrendLSTM | 0.453938253155 | 0.437649142861 |
| KalmanLSTM | 0.422577365003 | 0.408885350959 |

The D3 tests passed their fold-metric, ranking, training-history, alignment, and provenance assertions. Ranking RMSE is the arithmetic mean of fold RMSE values, not necessarily the pooled RMSE across observations. The candidate scores above are internal validation values from test fixtures, **not** protected-future performance measurements.

### 4.4 Failure accounting (M20-C4-D3)

Each family passed an audited failure scenario that retained **4 reserved epoch-folds**. This supports the contract that failed proposals retain their reserved search cost. Reserved computational budget must not be conflated with actual completed training epochs, particularly when training fails or stops early.

## 5. Warnings and limitations

The consolidated D4 run reported **18 non-blocking Keras warnings**, all associated with the GRU implementation's Sequential-layer `input_shape` / `input_dim` usage:

| Test module | Warning count |
| --- | ---: |
| GRU real-path integration (B2) | 10 |
| Reproducibility (D2) | 4 |
| Budget/evidence consistency (D3) | 4 |
| **Total** | **18** |

The warning recommends an explicit `Input(shape)` object as the first Sequential-model layer. Treat this as deferred API modernization, not an acceptance blocker. TensorFlow also emitted informational oneDNN/CPU messages in individual runs; these are distinct from the 18 pytest warnings in D4.

Other boundaries:

- No claim of forecasting accuracy on protected future observations or general model-family superiority.
- No claim of bitwise or cross-platform TensorFlow determinism.
- No claim that actual epochs must always equal reserved epoch-folds.
- Acceptance is limited to the tested authorization, validation, model and search configurations.
- Broader M20 reproducibility and diagnostic reporting remains in the subsequent M20-D workstream.

## 6. Version-control checkpoints

The following focused commits were reported on local branch `main`:

| Commit | Scope |
| --- | --- |
| `d566596` | Accepted LSTM and GRU integration tests (two files) |
| `92b06d4` | Accepted LinearTrendLSTM and KalmanLSTM integration tests (two files; 438 added lines) |
| `a10f42c` | Accepted D2 reproducibility and D3 budget/evidence tests (two files; 239 added lines) |

The reported staging checks for the latter two commits completed without whitespace errors. LF-to-CRLF notices were non-blocking. **This acceptance record is a new document and is not included in those existing commits.** GitHub synchronization is not asserted.

## 7. Acceptance decision and follow-up

**Decision: M20-C4 is ACCEPTED at the software-integration level**, based on the passing consolidated 28-test regression and the recorded D1–D3 evidence. The acceptance does not expand the verified scope beyond the tested configurations.

**Administrative follow-up:** Review and commit this document as `docs/milestones/M20_C4_Final_Acceptance_Record.md` using a focused documentation-only Git checkpoint.

**Next workstream:** M20-D — Reproducibility and Diagnostics. Begin with inspection of existing optimization execution, search-state, and diagnostic interfaces to avoid duplicating M20-C4 verification.
