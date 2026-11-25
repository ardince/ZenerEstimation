# M19-D4-C2 — Adversarial Verification Acceptance Record

**Project:** ZenerEstimation
**Milestone:** M19 — Selection Assessment
**Work package:** M19-D4-C2 — Integrated Adversarial Verification
**Checkpoint:** M19-D4-C2-D1 — Acceptance Record
**Decision:** ACCEPTED at software-contract verification level
**Record date:** 2026-11-25
**Evidence basis:** User-reported local pytest outputs and scoped Git status

## 1. Purpose and scope

M19-D4-C2 evaluates whether the existing internal ranking, observed-evidence integration, generalization authorization, and consolidated assessment contracts retain their scientific boundaries when confronted with invalid, incomplete, contradictory, or tampered inputs.

This work is verification-only. It does not authorize model selection, retraining, optimization, prospective certification, protected-holdout access, or scientific independence claims.

## 2. Verification coverage

The dedicated test module is `tests/test_m19_d4_c2_b1_adversarial.py`, covering 16 adversarial scenarios in five groups:

| Group | Cases | Contract concern |
|---|---:|---|
| A — Internal evidence integrity | A01–A04 | Invalid RMSE, non-finite metrics, forecast-calendar inconsistencies, and ranking-profile discrepancies |
| B — Observed evidence integrity | B01–B04 | Declaration binding, observation coverage, observed metric consistency, and temporal incompleteness |
| C — Authorization integrity | C01–C04 | Contradictory decisions, forged permissions, unauthorized promotion, and comparability claims |
| D — Scientific interpretation boundaries | D01–D03 | Internal recommendation versus selection authority, dataset labels versus authenticated provenance, and unavailable metrics |
| E — Reproducibility | E01 | Deterministic output and detached, non-mutating evidence handling |

These case descriptions identify the intended verification areas; the successful suite establishes the assertions encoded in the tests, not independent validation of every possible attack variant.

## 3. Execution evidence

### 3.1 Isolated B02 verification

Command:

```powershell
python -m pytest tests/test_m19_d4_c2_b1_adversarial.py -k "test_b02" -q
```

Reported outcome: **1 passed, 15 deselected in 1.42s**.

### 3.2 Focused C2-B1 verification

Command:

```powershell
python -m pytest tests/test_m19_d4_c2_b1_adversarial.py -q
```

Reported outcome: **16 passed in 1.47s**.

### 3.3 Integrated regression

The combined run included:

- `tests/test_m19_d4_c2_b1_adversarial.py`
- `tests/test_m19_d4_c1_b2_cross_contract.py`
- `tests/test_m19_d4_b2_consolidated_assessment.py`
- `tests/test_m19_d4_b2_r2_real_path.py`
- `tests/test_m19_d1_b2_b2_authorization_integration.py`
- `tests/test_m19_d1_c1_gap_assessment.py`
- `tests/test_m19_d1_c1_real_chain_integration.py`
- `tests/test_m19_b2_i2_c3_evidence_integration.py`
- `tests/test_m19_b2_i2_c3_m18_real_path.py`

Reported outcome: **136 passed in 4.11s**, with no failures or errors. The 136 includes the 16 focused C2-B1 tests; these are not additive counts.

## 4. B02 defect triage and resolution

B02 initially attempted to change `ObservedAlignmentResult.coverage_mask` using `dataclasses.replace`. This failed because `coverage_mask` is derived rather than an initialization field.

A subsequent test revision changed `ObservedAlignmentRow.eligible` alone. Its constructor rejected the invalid row because eligibility must agree with the presence of `observed_microVolt`.

The final revision constructed an internally valid excluded row by changing eligibility, observed value, and exclusion reasons together while retaining the original observability report. This created a contradiction between alignment-derived coverage and the separately supplied observability evidence. The isolated B02 test and full suite then passed.

**Classification:** Test-fixture construction defects; no production-code defect established by these failures. The current B02 assertion checks for `ValueError` but does not yet constrain the exception message or reason code to the specific coverage-consistency path. This is a non-blocking test-strengthening opportunity, not proof of a production issue.

## 5. Scientific and authorization boundaries

Acceptance of C2 confirms only the software behavior exercised by the test suite. It does **not** establish:

- Independent authenticity or external provenance of supplied evidence.
- Prospective scientific certification or complete temporal verification.
- Authorization to read, compare, rank, or disclose protected-holdout observations.
- Automatic model-selection authority from an internal ranking recommendation.
- Cross-track authentication merely because dataset identifiers match.
- Execution of controlled optimization, retraining, or operational forecasting.

Observed-only accuracy, internal ranking stability, and authorization decisions remain separate evidentiary tracks. Missing or incomplete evidence must not be promoted into a favorable scientific finding.

## 6. Repository checkpoint

The reported scoped Git status showed these untracked files:

```text
?? tests/test_m19_d4_b2_consolidated_assessment.py
?? tests/test_m19_d4_b2_r2_real_path.py
?? tests/test_m19_d4_c1_b2_cross_contract.py
?? tests/test_m19_d4_c2_b1_adversarial.py
?? zenerestimation/assessment/consolidated_assessment.py
```

This was a scoped status query, not an exhaustive repository status. No staging, commit, or push is authorized by this record. Unrelated working-tree changes and untracked files must remain untouched.

## 7. Acceptance decision

| Checkpoint | Decision | Basis |
|---|---|---|
| C2-A1 — Adversarial test design | ACCEPTED | Defined 16-scenario verification matrix |
| C2-B1 — Focused adversarial tests | ACCEPTED | 16/16 passed |
| C2-C1 — Integrated regression | ACCEPTED | 136/136 passed |
| C2-D1 — Acceptance record | DOCUMENTED | This record; scientific limitations explicitly preserved |

**Final C2 decision:** ACCEPTED at software-contract level, subject to the limitations above. **M19-D4 as a whole remains OPEN** pending final integrated acceptance review, documentation reconciliation, and separately authorized controlled Git operations.

## 8. Next checkpoint

Proceed to M19-D4 final integrated acceptance review. Confirm registered architecture, release, and milestone documentation, verify the intended repository file set, and only then prepare a narrowly scoped commit plan. Do not commence M20 without explicit authorization.
