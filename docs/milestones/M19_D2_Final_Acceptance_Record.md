# M19-D2 Final Acceptance Record

**Project:** ZenerEstimation
**Milestone:** M19-D2 — Internal Fold and Ranking Sensitivity
**Record date:** 2026-11-24
**Disposition:** Software-contract implementation, regression, and documentation review accepted; controlled commit pending.

## 1. Scope and delivered contracts

M19-D2 provides retrospective sensitivity diagnostics derived from M18-verified **internal expanding-window** forecast evidence. It does not retrain models, use protected external holdout observations, authorize model selection, or certify future performance.

| Section | Implementation | Contract |
| --- | --- | --- |
| D2-A | Source/interface inspection | Identified existing M18 evidence and validation boundaries |
| D2-B1 | `zenerestimation/optimization/internal_ranking.py` | Extracts verified internal fold RMSE values from M18-validated results |
| D2-B2 | `zenerestimation/assessment/fold_sensitivity.py` | Computes per-model leave-one-fold-out (LOFO) score changes |
| D2-C1 | `zenerestimation/assessment/ranking_sensitivity.py` | Compares cross-model ranks after omitting the same fold for every model |
| D2-C2 | `zenerestimation/assessment/ranking_stability.py` | Aggregates rank-change and winner-set-change frequencies, fractions, and rank displacement |
| D2-D | `tests/test_m19_d2_d_cross_contract.py` | Verifies cross-contract consistency and evidence-boundary behavior |

## 2. Evidence and scoring conventions

- Ranking RMSE is the arithmetic mean of verified fold RMSE values; it is distinct from pooled observation-level RMSE.
- LOFO diagnostics reaggregate existing verified folds; no refitting or forecast recomputation is performed.
- All models are compared using a common fold calendar and the same omitted fold index.
- Ranking uses M18-style exact ties and competition ranks (e.g., 1, 1, 3); near-equal floating-point scores are not silently merged.
- Invalid, incomplete, mismatched, or tampered evidence fails closed; insufficient-fold/model cases are reported as unavailable, not as stable.
- Outputs are deterministic and read-only relative to input results and independently supplied validation context.
- Stability fractions are descriptive proportions of internal fold omissions, **not** probabilities of future superiority.

## 3. Verification evidence

| Check | Observed result | Disposition |
| --- | --- | --- |
| D2-C1 focused | 15 passed, exit 0 | Passed |
| D2-C1 selected regression | 127 passed, exit 0 | Passed |
| D2-C2 focused | 25 passed, exit 0 | Passed |
| D2-C2 selected regression | 152 passed, exit 0 | Passed |
| D2-D cross-contract focused | 18 passed, exit 0 | Passed |
| D2-D selected combined regression | 170 passed, exit 0 | Passed |
| Full repository: `python -m pytest tests -q` | **3,560 passed, 161 warnings**, 1664.57 s (27m 44s), exit 0 | Passed with warnings |
| `git diff --check` | No whitespace errors reported | Passed |

The full-suite warnings were reported in existing forecasting/optimization tests, including statsmodels/NumPy degenerate diagnostic calculations, ARIMA initialization/convergence, and Keras Sequential input-shape usage. They were not reported as D2 test failures. Their novelty relative to prior runs has not been established; do not silently suppress them.

## 4. Source and working-tree inventory

**D2 production changes:**

- Modified: `zenerestimation/optimization/internal_ranking.py`
- New: `zenerestimation/assessment/fold_sensitivity.py`
- New: `zenerestimation/assessment/ranking_sensitivity.py`
- New: `zenerestimation/assessment/ranking_stability.py`

**D2 test additions:**

- `tests/test_m19_d2_b1_verified_folds.py`
- `tests/test_m19_d2_b2_fold_sensitivity.py`
- `tests/test_m19_d2_c1_ranking_sensitivity.py`
- `tests/test_m19_d2_c2_ranking_stability.py`
- `tests/test_m19_d2_d_cross_contract.py`

At inventory time, `internal_ranking.py` was modified and the other eight D2 source/test files were untracked. The repository also contained unrelated modified and untracked files, including `docs/draft/development_flow.docx`; these are **out of scope** and must not be included by broad staging.

## 5. Limitations and acceptance boundaries

This record accepts the reported software-contract tests and their integration behavior. It does **not** claim independent source authentication, external scientific validation, protected-holdout performance, prospective ranking stability, or model-selection authority. Insufficient-fold and insufficient-model conditions are covered by contract tests. The D2-D integration suite additionally verifies that one-model unavailability propagates through the real assessment chain without fabricating ranking-stability statistics.

## 6. Remaining closure actions

- [x] Compare wording/format with existing `M19_D1_Final_Acceptance_Record.md` and `M19_B2_I2_Final_Acceptance_Record.md` (reviewed 2026-11-24).
- [x] Register and review the prepared D2 addenda in `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT_HISTORY.md`, and `docs/RELEASE_NOTES.md`; only headings of the repository files have been inspected, so no complete-file reconciliation is claimed.
- [x] Confirm the final D2-only file inventory and `git diff --check`.
- [x] Stage only approved D2 implementation, tests, and documentation files; exclude unrelated changes.
- [ ] Create a controlled commit after documentation review; record the commit identifier here.

**Commit identifier:** Pending
**Documentation closure:** Reviewed and staged
**Overall M19-D2 status:** Implementation and regression accepted; formal repository closure pending.
