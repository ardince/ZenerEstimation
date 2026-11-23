# M19-D1-C1 — Software-contract acceptance checkpoint

**Project:** ZenerEstimation
**Checkpoint date:** 2026-11-23
**Disposition:** **Accepted at the focused software-contract level**; full-repository regression and D1-D review remain pending.
**Git status:** This document records test evidence; it does not represent a commit, tag, or remote synchronization.

## 1. Scope

M19-D1-C1 introduces a **read-only, fail-closed generalization-gap availability assessment**. It delegates existing internal-evidence and independent-benchmark authorization verification to the accepted D1-B2-B2 reconciliation contract. It does **not** assess protected-holdout targets, compute a numerical gap, verify scientific comparability, or grant protected-holdout assessment, comparison, ranking, or disclosure permissions.

**Production module:** `zenerestimation/comparison/generalization_gap_assessment.py`
**Entry point:** `assess_generalization_gap(...)`
**Assessment version:** `m19_d1_c1_gap_assessment_v1`

The assessment accepts the existing internal selection artifact, results, validation context, C2 benchmark authorization inputs, and optional supplied authorization decision. It recomputes through `reconcile_generalization_authorization()` rather than treating a supplied decision as authority.

## 2. Contract and invariants

- Only upstream `AUTHORIZATION_REQUIRED` and `AUTHORIZATION_REJECTED` decisions are accepted; the status is preserved.
- Verified internal selection identity (`selected_model`, `selection_decision_digest`) must be present; contradictory upstream evidence or unexpected permissions raise an error.
- `internal_evidence_verified` must be `True`; `holdout_assessed` must be `False`.
- `generalization_gap` and `generalization_gap_metric` remain `None`.
- `comparability_verified` remains `False`: a matching model name or benchmark scope digest is not a verified selected-model-to-benchmark participant binding.
- `protected_holdout_assessment_permitted`, `protected_holdout_comparison_permitted`, `protected_holdout_ranking_permitted`, and `protected_holdout_disclosure_permitted` remain `False`.
- Supplied-decision mismatches propagate a reduced rejection record; absent benchmark digests are not invented.
- Output is deterministic and detached from mutable upstream records. No model execution or holdout target disclosure is introduced by the facade.

## 3. User-executed test evidence

**Environment:** Windows PowerShell; Conda `zener`; Python 3.11.15; pytest 9.1.1; pluggy 1.6.0; anyio 4.15.1.

| Run | Command / scope | Result |
| --- | --- | --- |
| D1-C1 isolated contract | `python -m pytest tests/test_m19_d1_c1_gap_assessment.py -v` | **17 passed** |
| D1 combined regression before new integration file | Five prior D1 test files plus D1-C1 isolated tests | **67 passed** |
| D1-C1 real-chain integration | `python -m pytest tests/test_m19_d1_c1_real_chain_integration.py -v` | **5 passed in 1.44 s** |
| D1 combined regression after integration | Seven D1 test files, including both D1-C1 files | **72 passed in 3.11 s** |

The five integration tests are **included** in the 72-test regression, not additional to it. The 17 isolated tests are likewise included. The 67-test run is superseded by the 72-test combined run for the focused D1 baseline.

**D1-C1 tests:**

- `tests/test_m19_d1_c1_gap_assessment.py`
- `tests/test_m19_d1_c1_real_chain_integration.py`

**Other D1 files in the 72-test run:**

- `tests/test_m19_d1_b1_internal_evidence.py`
- `tests/test_m19_d1_b1_real_artifact_integration.py`
- `tests/test_m19_d1_b2_b1_holdout_boundary.py`
- `tests/test_m19_d1_b2_b2_authorization_integration.py`
- `tests/test_m19_d1_b2_b3_downstream_separation.py`

## 4. Integration evidence and limits

The five real-chain integration tests verify: (1) actual C2 rejection propagation through the D1 chain; (2) a **controlled, insufficient C2 decision** passed through real D1 reconciliation and assessment; (3) supplied-decision mismatch rejection with a reduced record; (4) tampered internal-selection rejection; and (5) deterministic read-only behavior.

**Important qualification:** The controlled authorization-required test substitutes a C2 decision; it does **not** prove a fully real, independently authorized C2 chain. The real C2 rejection test exercises the actual C2 function using deliberately invalid C2 inputs. None of these tests establishes independently trusted chronology, provenance, access completeness, prospective authorization, or model-to-benchmark comparability. Test results establish examined software interfaces, not repository-wide access-control guarantees.

## 5. Relationship to prior acceptance

The earlier `M19_D1_B1_B2_ACCEPTANCE_CHECKPOINT.md` recorded accepted D1-B1/B2 contracts and a full-repository run of **3,445 passed, 161 warnings, 1,520.87 s** on 2026-11-20. That result is a **historical baseline**, not a full-suite run after D1-C1. The new 72-test D1 regression is the current focused evidence.

Existing single-model B1–B4 holdout eligibility, disclosure, scoring, and evidence contracts remain conceptually separate from D1-C1 and from independent cross-model authorization. Ranking RMSE (mean fold RMSE) must not be conflated with pooled RMSE.

## 6. Acceptance decision and remaining work

**Decision:** Accept **M19-D1-C1 as a fail-closed software-contract implementation**, based on 17 isolated tests, five integration tests, and the 72-test combined D1 regression. This is **not** acceptance of numerical generalization-gap estimation or protected-holdout authorization.

**Open items:**

1. Perform D1-D downstream/adversarial integration review, including disclosure paths, identity binding, and unexpected cross-contract permissions.
2. Keep any proposed D1-C2 numerical-gap computation **deferred** pending independently reviewed authorization, explicit selected-model/dataset/date/metric binding, and scientifically defensible comparability.
3. Run the full repository test suite after subsequent changes and record the actual outcome before final D1 milestone acceptance.
4. Update `docs/ARCHITECTURE.md` and `docs/DEVELOPMENT_HISTORY.md` with this qualified acceptance after reviewing the authoritative roadmap wording.
5. Review Git status and stage only explicitly approved paths; **do not** use `git add .` or imply a remote sync.

## 7. Reproducibility commands

From the repository root in the `zener` environment:

```powershell
$D1Tests = @(
    "tests/test_m19_d1_b1_internal_evidence.py"
    "tests/test_m19_d1_b1_real_artifact_integration.py"
    "tests/test_m19_d1_b2_b1_holdout_boundary.py"
    "tests/test_m19_d1_b2_b2_authorization_integration.py"
    "tests/test_m19_d1_b2_b3_downstream_separation.py"
    "tests/test_m19_d1_c1_gap_assessment.py"
    "tests/test_m19_d1_c1_real_chain_integration.py"
)
python -m pytest $D1Tests -q
```

**Expected historical result for the exact reported run:** `72 passed in 3.11s`. Future runs may differ as the repository evolves.
