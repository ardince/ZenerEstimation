# M19-D1-D — Downstream Integration and Adversarial Review

**Acceptance checkpoint**
**Date:** 2026-11-23
**Project:** ZenerEstimation
**Disposition:** **ACCEPTED — tested fail-closed software-contract scope only**
**Scientific authorization:** **NOT ESTABLISHED**

## 1. Purpose and scope

M19-D1-D examines whether downstream single-model protected-holdout scoring and evaluation evidence can be improperly promoted into authorization for the D1 generalization-gap assessment. It builds on the previously accepted M19-D1-C1 fail-closed assessment, and does not introduce a numerical generalization-gap computation pathway.

The D1-D1 through D1-D4 subdivisions are the working implementation plan used in this review; this checkpoint does not assert that those identifiers are part of an independently verified official roadmap.

## 2. Inspected contracts (D1-D1)

- `zenerestimation/assessment/holdout_error_scoring.py`: `score_disclosed_holdout()` scores only already-disclosed, declaration-bound observations. It validates eligibility, disclosure schema and digest, dates, provenance, and target alignment. A `HOLDOUT_SCORED` record retains `scientific_independence = "not_established"`.
- `zenerestimation/assessment/holdout_evaluation_evidence.py`: `build_holdout_evaluation_evidence()` recomputes scoring and compares the complete canonical record. The resulting evidence retains `scientific_independence = "not_established"` and `authentication = "not_established"`.
- `tests/test_m19_d1_b2_b3_downstream_separation.py`: established separation tests ensure internal-selection evidence is not treated as B1 eligibility or as C2 authorization.
- `zenerestimation/comparison/generalization_gap_assessment.py`: D1-C1 delegates to D1-B2-B2 authorization reconciliation, denies all protected-holdout permissions, reports no numerical gap, and does not certify comparability.

**Boundary:** Single-model scoring or verified evaluation evidence is not independent cross-model benchmark authorization. A digest establishes internal consistency, not external authentication or independent scientific provenance.

## 3. Adversarial tests (D1-D2)

**Test file:** `tests/test_m19_d1_d2_downstream_adversarial.py`
**Reported command:**

```powershell
python -m pytest tests/test_m19_d1_d2_downstream_adversarial.py -v
```

**Reported outcome:** **14 passed in 1.47 s**, zero failures/errors.

Coverage demonstrated by the named passing tests:

1. Real C2 rejection cannot be overridden by B4 evidence.
2. B3 scoring, B2 disclosure, and B1 eligibility snapshots cannot be supplied as C2 authorization decisions.
3. Five parameterized direct-injection cases reject holdout/permission-related arguments at the D1 API.
4. Verified B4 evidence does not establish D1 comparability.
5. Matching selected-model names do not establish benchmark binding.
6. D1 assessment does not call B3 scoring or B4 evidence construction.
7. The tested D1 path does not mutate the supplied B3/B4 fixture chain.
8. B3 scoring rejects internal-selection evidence used as its eligibility snapshot.

### Test-development corrections

- An initial run reported seven setup errors because the imported `chain` fixture depended on an unimported `case` fixture. The missing fixture import was added.
- A subsequent run reported one test failure because `deepcopy(chain)` attempted to copy an immutable `mappingproxy` inside `AssessmentDeclaration`. The test was corrected to preserve the immutable declaration by reference and deep-copy the other snapshot values.
- After these **test-only corrections**, the user reported 14/14 passing. Neither error established a production defect, and no production change was required to address them.

## 4. Combined integration regression (D1-D3)

The user executed the following eight-file suite:

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

**Reported outcome:** **86 passed**, zero failures/errors.

The 86-test total **includes** the 14 D1-D2 adversarial tests and the 17 D1-C1 isolated contract tests and five D1-C1 real-chain tests. These counts must not be added to the 86-test total.

### Prior accepted baseline

- D1-C1 isolated contract suite: **17 passed**.
- D1-C1 real-chain integration suite: **5 passed**.
- Prior D1 combined regression: **72 passed**.
- Prior complete-project baseline: **3,445 passed** (before the current D1-D additions). **A new full-project regression has not been reported.**

All execution outcomes above are taken from the user's Windows PowerShell / Conda `zener` pytest logs; this checkpoint does not claim independent execution in the assistant environment.

## 5. Acceptance decision (D1-D4)

**Accepted:** The tested D1-D downstream-separation and adversarial contracts, including real C2 rejection propagation, prevention of B1/B2/B3/B4 evidence promotion, direct-input injection rejection, and read-only behavior, with an 86-test combined D1 regression passing.

**Not accepted or established:**

- Independent scientific authorization, external authenticity, or trusted real-world chronology.
- Permission to access, disclose, compare, or rank protected-holdout observations via D1.
- Verified selected-model-to-benchmark participant identity binding, dataset/date/horizon/metric comparability, or a numerical generalization gap.
- A full-project regression result following D1-D changes.
- Completion of the entire M19 milestone or authorization of a future D1-C2 numerical engine.

The C2 authorization-required/rejected decisions remain fail-closed. Test-issued evidence, verified digests, and B4 single-model evaluation records must not be interpreted as scientific authorization.

## 6. Follow-up and repository handling

1. Record this checkpoint under `docs/milestones/M19_D1_D_Acceptance_Checkpoint.md` after local review.
2. Update `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT_HISTORY.md`, and release notes as appropriate, preserving the scientific-authorization caveat.
3. Perform a full-project pytest regression before broader D1 or M19 final acceptance; compare with the prior 3,445-pass baseline without assuming the new total.
4. Review any future numerical generalization-gap design as a separate authorization and identity/comparability contract. No implicit permission upgrade is allowed.
5. Inspect Git changes explicitly; do not stage unrelated data, backups, or outputs. No commit, push, or GitHub synchronization is claimed here.

## 7. Acceptance summary

**M19-D1-D: ACCEPTED at tested software-contract scope.**
**Combined D1 regression: 86 passed.**
**Numerical generalization gap: not computed; authorization not established.**
**Full-project post-D1-D regression: pending.**
