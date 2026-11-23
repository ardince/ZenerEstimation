# ZenerEstimation — M19-B2-I2 Final Acceptance Record

**Document ID:** M19-B2-I2-ACCEPTANCE-001
**Date:** 18 November 2026
**Scope:** M19-B2-I2-A through I2-D
**Decision:** ACCEPTED — software-contract validation
**Scientific certification:** NOT GRANTED
**Release status:** Local development acceptance; no new numbered release or remote publication asserted.

## 1. Decision and scope

M19-B2-I2 is accepted at the **software-contract validation** level. The accepted scope consists of temporal eligibility, target observability, verified forecast alignment, observed-only scoring, integrated evidence, and cross-contract adversarial testing. The decision is grounded in user-supplied pytest outputs; it is not an independent repository audit or scientific certification.

## 2. Accepted contracts

| Section | Implementation | Tested responsibility | Passing tests |
| --- | --- | --- | ---: |
| I2-A | `zenerestimation/assessment/temporal.py` | M18 plan/execution binding; cutoff, calendar and evidence-status handling | 19 |
| I2-B | `zenerestimation/assessment/observability.py` | Measurement provenance, missing quarters, eligibility and coverage | 30 |
| I2-C1 | `zenerestimation/assessment/observed_scoring.py` | Verified M18 forecast alignment; complete horizon and observation mask | 19 |
| I2-C2 | `zenerestimation/assessment/observed_metrics.py` | Observed-only RMSE/MAE/MAPE; zero and missing actuals | 11 |
| I2-C3 | `zenerestimation/assessment/evidence_integration.py` | Cross-contract identity, digest, calendar, coverage, metric and snapshot consistency | 50 |
| I2-D | `tests/test_m19_b2_i2_d_cross_contract.py` | Coordinated changes, temporal precedence, invalid M18 evidence, metric denominators and mutation isolation | 7 |
| **I2 combined regression** | **9 test files** | **Single combined I2 invocation** | **136** |

## 3. Compatibility evidence

A separate targeted M18 pytest invocation covered:

- `tests/test_m18_forecast_plan_contract.py`
- `tests/test_m18_forecast_execution_contract.py`

**Result:** 89 passed, zero reported failures/errors.

**Total reported evidence:** 136 I2 tests plus 89 targeted M18 tests = **225 passing test cases across two separate invocations**. This is not a single 225-test run and is not a repository-wide regression claim.

**Environment reported:** Windows; Python 3.11.15; pytest 9.1.1; conda environment `zener`; `pyproject.toml`.

## 4. Cross-contract acceptance findings

1. Coordinated changes to observation evidence can remain internally consistent; internal consistency is not independent measurement authentication.
2. Even when all supplied temporal findings are verified, the summary `verified_findings` does not constitute independent scientific certification.
3. Invalid real M18 execution evidence is rejected before downstream integration in the tested path.
4. Complete observation coverage does not override UNKNOWN or FAILED temporal evidence.
5. Zero-valued actuals remain eligible for RMSE/MAE while being excluded from MAPE's nonzero-actual denominator; missing targets are not scored.
6. Integrated serialized snapshots remain detached from subsequent mutations of nested source objects in the tested cases.

## 5. Scientific and operational limitations

- Source measurements, their provenance and chronology have **not** been independently authenticated by these tests.
- Verified findings do **not** establish prospective forecasting validity, leakage absence in unexamined workflows, or external scientific certification.
- Availability of observed-only metrics does **not** supersede failed or incomplete temporal eligibility.
- The targeted M18 regression covers two contract test files, not the full M18 milestone or entire repository.
- No Git commit, GitHub push, release number change, or external publication is implied by this acceptance record.

## 6. Handoff conditions

Downstream M19 sections must preserve evidence-status distinctions, observed-only metric eligibility and denominator rules, M18 execution-artifact binding, and the difference between verified software consistency and independent scientific authentication. The 136-test I2 suite is the local regression baseline for changes affecting these contracts.

## 7. Final disposition

**ACCEPTED — M19-B2-I2 software-contract validation.**
**Scientific certification:** NOT GRANTED.
**Outstanding:** Broader repository regression and any independent scientific/provenance certification required by subsequent work.
