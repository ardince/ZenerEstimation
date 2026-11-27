# M20-D Final Acceptance Record

**Project:** ZenerEstimation
**Scope:** Milestone 20, Section D â€” Reproducibility and Diagnostics (D-B1 through D-B4)
**Evidence date:** 27 November 2026
**Disposition:** Accepted at targeted software-contract verification level
**Framework version:** 0.7.0 (unchanged)

## 1. Scope and implementation

| Component | Accepted scope | Local commit |
| --- | --- | --- |
| M20-D-B1 | Independent Grid Search replay verification | `933609c` |
| M20-D-B2 | Independent seeded Random Search replay verification | `594f551` |
| M20-D-B3 | Independent adaptive Bayesian Search replay verification | `96d6d35` |
| M20-D-B4 | Cross-strategy budget/termination hardening and additive diagnostics | `69f620e` |

Implementation: `zenerestimation/optimization/search_replay_m20.py`. Dedicated tests: `tests/test_m20_d_b1_grid_replay.py`, `tests/test_m20_d_b2_random_replay.py`, `tests/test_m20_d_b3_bayesian_replay.py`, `tests/test_m20_d_b4_cross_strategy_diagnostics.py`.

The replay verifier checks search proposal identity and order, feedback consistency, deterministic selection, budget accounting and termination evidence against supplied typed execution artifacts. B4 specifically reconstructs the next adaptive Bayesian candidate for an epoch-fold budget stop rather than substituting a sequential grid candidate. Additive diagnostics report proposal/evaluation/failure counts, pending sequence, stop reason, trial/epoch-fold limits and reserved epoch-fold consumption. Replay is read-only; it does not train models or consume grants.

## 2. Reported validation evidence

| Run | Result | Notes |
| --- | --- | --- |
| B4 focused | 13 passed | New B4 cross-strategy tests |
| B1â€“B4 combined | 71 passed, 33 warnings | Regression compatibility |
| Final eleven-module targeted acceptance | **106 passed, 48 warnings, zero failures** | Includes replay, real ARIMA/Kalman search, Bayesian adversarial and authorized execution integration |

The runs overlap; test counts must not be added together. Results were reported from the local Windows `zener` environment. The final run was **not** a full-repository test invocation.

### Final consolidated pytest invocation

```powershell
python -m pytest `
    tests/test_m20_d_b1_grid_replay.py `
    tests/test_m20_d_b2_random_replay.py `
    tests/test_m20_d_b3_bayesian_replay.py `
    tests/test_m20_d_b4_cross_strategy_diagnostics.py `
    tests/test_m20_c1_b_arima_grid_real_path.py `
    tests/test_m20_c2_b_arima_random_real_path.py `
    tests/test_m20_c2_c_kalman_random_real_path.py `
    tests/test_m20_c3_b_arima_bayesian_real_path.py `
    tests/test_m20_c3_c_kalman_bayesian_real_path.py `
    tests/test_m20_c3_d_bayesian_numerical_adversarial.py `
    tests/test_m20_b5_e_real_path.py `
    -q
```

Reported outcome: `106 passed, 48 warnings`. Of the warnings, 33 arose in the D-B3 Bayesian replay tests and 15 in C3-B ARIMA Bayesian real-path tests. They were `statsmodels` SARIMAX notices that non-stationary starting autoregressive parameters were found and zero starting parameters were used. No numerical-warning remediation is claimed.

## 3. Git evidence

The user reported the B4 commit as `69f620e (HEAD -> main)` with two files changed, 145 insertions and two deletions; `git diff --cached --check` returned no whitespace errors, and the committed files had no subsequent short-status output. B1â€“B3 commits are listed in Section 1. These facts support local implementation-commit acceptance; they do not establish a remote push or clean status for the entire repository.

## 4. Trust and scientific boundaries

- Independent replay verifies internal consistency of the supplied search/execution evidence; it does not independently authenticate source measurements, chronology or external provenance.
- Authorization is process-local and single-use within a trusted workflow, not an adversarial security boundary against arbitrary in-process code.
- Development-only ranking is not protected-holdout authorization or scientific validation of prospective forecasting accuracy.
- Synthetic neural-budget tests validate generic resource accounting, not full real neural/hybrid optimization integration.
- No post-D full-repository regression, new framework release, GitHub publication or entire Milestone 20 acceptance is asserted.

## 5. Acceptance decision and remaining work

**Decision:** M20-D-B1â€“B4 accepted as complete at the tested software-contract level, based on the reported local commits and 106-test targeted cross-contract suite. Documentation changes in this record are proposed for separate review and Git registration.

**Remaining:** M20-C4 neural/hybrid integration and M20-E milestone-wide acceptance remain separate workstreams. Any broader Milestone 20 status should be reconciled against its authoritative plan before closure.
