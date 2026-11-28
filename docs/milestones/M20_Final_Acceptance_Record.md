# Milestone 20 — Final Software-Contract Acceptance Record

**Project:** ZenerEstimation
**Milestone:** M20 — Authorized Model Optimization, Integration, Reproducibility and Diagnostics
**Evidence recorded:** 28 November 2026
**Disposition:** Accepted at the tested software-contract level, subject to the explicit scientific and operational limitations below
**Framework version:** 0.7.0 (unchanged; unreleased milestone checkpoint)
**Local Git baseline:** `54bc080` (`main`), prior to this final documentation update

## 1. Scope and acceptance basis

This record consolidates the previously accepted M20-A/C1, M20-C4 and M20-D work with the implementation/test evidence for M20-B2–B5 and M20-C2/C3. M20-E1 established the acceptance and tracked-test inventory; M20-E2 and M20-E3 passed consolidated and repository-wide regressions. This is a milestone-wide **software-contract** acceptance, not a claim of scientific certification or deployment readiness.

## 2. Implemented scope and traceability

| Scope | Contract or verification | Local commit evidence |
| --- | --- | --- |
| M20-A / C1 | Architecture review; real ARIMA/Kalman bounded Grid Search; alignment and failure-path tests | `2b770af` |
| M20-B2 | Process-local, single-use local optimization authorization | `f065258` |
| M20-B3 | Development-only dataset binding | `12aff14` |
| M20-B4 | Bounded Grid, Random and Bayesian search and adversarial hardening | `e02aacd` |
| M20-B5 | Authorized execution and real-path/concurrency verification | `683b02c` |
| M20-C2 | Real-model authorized Random Search integration | `accd306` |
| M20-C3 | Real-model authorized Bayesian Search integration | `1e04a72` |
| M20-C4 | LSTM, GRU, LinearTrendLSTM and KalmanLSTM integration; reproducibility and budget evidence | `d566596`, `92b06d4`, `a10f42c`, `83b86de` |
| M20-D | Independent Grid, Random and Bayesian replay; cross-strategy diagnostics and hardening | `933609c`, `594f551`, `96d6d35`, `69f620e`, `54bc080` |
| M20-E | Inventory, consolidated M20 regression, full repository regression and this final acceptance documentation | User-reported local test output; documentation commit pending |

The dates and status of earlier checkpoints remain historical. M20-C4 was committed before M20-D; its prior omission from the M20-D checkpoint's list of completed work does not invalidate the actual C4 Git history.

## 3. Existing acceptance records

- `docs/milestones/M20_A_C1_Acceptance_Record.md`
- `docs/milestones/M20_C4_Final_Acceptance_Record.md`
- `docs/milestones/M20_D_Final_Acceptance_Record.md`

The current final record supplements, rather than rewrites, these earlier records. Separate standalone acceptance records for every intermediate B/C stage were not listed in the supplied M20-E1 inventory.

## 4. M20-E1 — Inventory evidence

- `git ls-files -- 'tests/test_m20_*.py'` returned **26 tracked M20 test modules**.
- The working directory also contained an untracked `tests/test_m20_c1_b_arima_grid_real_path_corrected.py`; it was deliberately excluded from the tracked-file test selection. Its retention/disposition remains a separate repository-hygiene matter.
- The M20-E1 inventory showed the three existing M20 acceptance records above.
- `origin/main` was at `32a99d0`, while local `main` was at `54bc080`; GitHub synchronization was not established.

## 5. M20-E2 — Consolidated tracked M20 regression

Executed locally in the `zener` environment:

```powershell
$M20Tests = @(git ls-files -- 'tests/test_m20_*.py')
python -m pytest @M20Tests -q
```

**Result:** 231 passed, 66 warnings, zero reported failures; **253.05 seconds** (4:13). This run covers the tracked M20-B/C/D test modules, including authorization, development binding, bounded execution, classical/neural/hybrid real paths, reproducibility, budget evidence and independent replay.

Warning categories: 48 `statsmodels` SARIMAX non-stationary autoregressive starting-parameter notices and 18 Keras Sequential RNN `input_shape`/`input_dim` configuration notices. Warnings are not described as fixed.

## 6. M20-E3 — Full repository regression

Executed locally:

```powershell
python -m pytest -q
```

**Result:** 3,951 passed, 227 warnings, zero reported failures; **927.19 seconds** (15:27). Compared with the previously documented pre-C1 full-suite baseline (3,813 passed, 161 warnings), this run reports **138 more passing tests and 66 more warnings**. The baseline comparison reflects different repository states and does not imply that every new test is exclusively attributable to M20.

Reported warning families include ARIMA/SARIMAX initialization and convergence, Keras Sequential RNN input configuration, and NumPy/Statsmodels zero-variance or degenerate residual diagnostics. No warning remediation or prospective model-quality improvement is claimed.

The 231-test M20 suite is a subset of the broader test inventory and **must not be added** to the 3,951-test repository result. Earlier focused regressions also overlap these runs.

## 7. Acceptance criteria and boundaries

**Accepted software contracts:** approved development-only search requests; bounded trials and epoch-fold accounting; single-use local authorization workflow; temporal fold alignment; model-specific integration; deterministic evidence and replay checks; failure/abort handling; and compatibility with the existing regression suite, to the extent exercised by the passing tests.

**Not established:** independent authentication of measurements or chronology; an adversarial security boundary against arbitrary code running in the same process; protected-holdout access or cross-model authorization; numerical external generalization-gap evidence; prospective forecasting superiority, stability or scientific certification; and production deployment readiness.

The distinction between **ranking RMSE** (arithmetic mean of fold RMSE) and **pooled RMSE** (across observations) remains intact. Frozen measured-holdout benchmarks and unvalidated operational extrapolations remain separate; no benchmark metric, model ranking, seed policy or historical evidence is revised by this acceptance record.

## 8. Outstanding maintenance and publication decisions

- Non-failing numerical-library and Keras warnings remain available for later targeted maintenance.
- The untracked corrected-copy test and other unrelated working-tree artifacts require separate selective review; no blanket staging or deletion is authorized.
- Framework version remains `0.7.0`; this record does not create a numbered release.
- No GitHub push is claimed. The local/remote branch difference must be reviewed before any future synchronization.
- The final documentation files require local installation, `git diff --cached --check`, staged review and a separate Git commit before their commit status can be asserted.

## 9. Final disposition

**Milestone 20 is accepted and closed at the tested software-contract level based on the user-reported M20-E2 and M20-E3 passing regressions.** Documentation integration and publication remain separate operational steps. No protected-holdout authorization, independent scientific certification, new framework release or GitHub synchronization is implied.
