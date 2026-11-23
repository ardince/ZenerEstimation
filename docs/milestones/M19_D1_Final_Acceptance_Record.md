# M19-D1 — Final Software-Contract Acceptance Record

**Project:** ZenerEstimation
**Date:** 2026-11-23
**Disposition:** **ACCEPTED — tested M19-D1 software-contract scope**
**Scientific authorization:** **NOT ESTABLISHED**
**Numerical generalization gap:** **NOT COMPUTED / NOT AUTHORIZED**
**Repository publication:** Pending explicit local Git audit, commit and push

## 1. Scope of acceptance

M19-D1-B1 verifies M18 internal selection and reconstructs M17 fold accuracy evidence. D1-B2-B1 and B2-B2 enforce the internal-to-protected-holdout authorization boundary and read-only C2 reconciliation. D1-B2-B3 protects separation from existing single-model holdout B1–B4 contracts. D1-C1 provides a fail-closed generalization-gap assessment facade; D1-D reviews downstream integration and adversarial bypass attempts.

This acceptance covers these implemented and tested contracts; it does not assert that the independently authoritative milestone roadmap, all M19 work, or a future numerical D1-C2 engine is complete.

## 2. Evidence and reproducibility

| Evidence | Result | Notes |
| --- | --- | --- |
| D1-C1 isolated contract tests | 17 passed | Historical focused run |
| D1-C1 real-chain integration | 5 passed | Includes real rejected C2 path; insufficient C2 path controlled |
| D1-D2 downstream adversarial | 14 passed in 1.47 s | After test-fixture-only corrections |
| D1 combined regression (8 files) | **86 passed** | Includes the 14 D1-D2 tests; do not double-count |
| Full repository pytest | **3,481 passed, 161 warnings** | 1,616.84 s (0:26:56), 2026-11-23 |
| Full repository exit code | **0** | Reported by user after local PowerShell run |
| Previous full-repository baseline | 3,445 passed, 161 warnings | 2026-11-20; increase of 36 passing tests |

The complete pytest log was supplied as `M19_D1_Full_Regression.log`. The assistant inspected its summary and warning categories; execution occurred in the user's local Windows `zener` environment, not in the assistant's container. The 161 warnings include statsmodels/NumPy numerical and ARIMA initialization/convergence warnings and Keras `input_shape` guidance. Warning count is unchanged relative to the prior baseline, but the warnings are not resolved.

Full regression command (from repository root, `zener` environment):

```powershell
python -m pytest -q *> M19_D1_Full_Regression.log
$PytestExitCode = $LASTEXITCODE
Get-Content M19_D1_Full_Regression.log -Tail 35
Write-Host "Pytest exit code: $PytestExitCode"
```

D1 combined regression files:

```powershell
$D1Tests = @(
    "tests/test_m19_d1_b1_internal_evidence.py"
    "tests/test_m19_d1_b1_real_artifact_integration.py"
    "tests/test_m19_d1_b2_b1_holdout_boundary.py"
    "tests/test_m19_d1_b2_b2_authorization_integration.py"
    "tests/test_m19_d1_b2_b3_downstream_separation.py"
    "tests/test_m19_d1_c1_gap_assessment.py"
    "tests/test_m19_d1_c1_real_chain_integration.py"
    "tests/test_m19_d1_d2_downstream_adversarial.py"
)
python -m pytest $D1Tests -q
```

## 3. Accepted safety and scientific boundaries

1. Internal model selection and fold accuracy do not establish external generalization or independent protected-holdout authorization.
2. Ranking RMSE (arithmetic mean of fold RMSE) is not pooled RMSE (square root of pooled squared error mean); preserve both definitions.
3. D1-C1 reports `holdout_assessed = False`, `generalization_gap = None`, `generalization_gap_metric = None`, and `comparability_verified = False`. Protected-holdout assessment, comparison, ranking, and disclosure permissions remain `False`.
4. Recomputed C2 `AUTHORIZATION_REQUIRED`/`AUTHORIZATION_REJECTED` decisions remain fail-closed; inconsistent supplied decisions are rejected.
5. Single-model B3 scoring and B4 evidence reconstruction do not confer independent cross-model authorization, verified participant identity binding, or authenticated chronology.
6. Deterministic digests and verified software evidence are not independent source authentication, prospective authorization, or proof of scientific comparability.

## 4. Non-claims and deferred work

- No independent scientific provenance, real-world chronology, access completeness, prospective authorization, or model-to-benchmark participant binding is established.
- No protected-holdout observations are newly authorized for D1 access, disclosure, ranking, or comparison.
- No numerical external generalization-gap value is accepted. A future numerical engine requires separate reviewed authorization, dataset/participant/date/horizon/metric binding, and comparability evidence.
- Passing tests validate examined contracts; they are not proof of repository-wide access-control guarantees.
- This acceptance is not a numbered release, a Git commit, a remote push, or complete Milestone 19 acceptance.

## 5. Documentation reconciliation

The accompanying `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT_HISTORY.md`, and `docs/RELEASE_NOTES.md` preserve the prior B2-I2 and D1-B1/B2 history while adding the final D1 contract acceptance and regression evidence. Prior checkpoints are retained as historical snapshots; their earlier statements that full regression was pending should be read as **true at their creation dates**, superseded by this final 2026-11-23 record.

**Prior records:** `docs/milestones/M19_D1_B1_B2_ACCEPTANCE_CHECKPOINT.md`, `M19_D1_C1_ACCEPTANCE_CHECKPOINT.md`, and `M19_D1_D_Acceptance_Checkpoint.md` (case-sensitive spelling may differ from local working tree; verify).

## 6. Git/GitHub synchronization gate

Before committing or pushing, inspect actual local Git status, branch/upstream, remote URL, tracked changes, and untracked source dependencies. Do not stage the entire tree or include datasets, logs, archives, secrets, or unrelated work by default. Inspect the staged diff and tests, then commit and push only after explicit approval of the exact file set and target branch. This document records **no Git action performed**.

**Final decision:** M19-D1 accepted at the tested software-contract level, supported by 86/86 D1 and 3,481/3,481 full-project passing tests. Independent scientific authorization and numerical generalization-gap computation remain deferred.
