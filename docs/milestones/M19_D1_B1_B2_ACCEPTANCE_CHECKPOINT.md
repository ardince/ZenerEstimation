# M19-D1-B1/B2 — Acceptance checkpoint

**Project:** ZenerEstimation
**Date:** 2026-11-20
**Status:** Focused and full-regression acceptance passed; local Git checkpoint not yet committed or synchronized.

## 1. Scope and accepted components

| Work package | Scope | Acceptance |
| --- | --- | --- |
| M19-D1-B1 | Re-verify internal generalization and selection evidence, without accessing protected-holdout targets | Accepted |
| M19-D1-B2-B1 | Establish internal-to-protected-holdout boundary with denied permissions | Accepted |
| M19-D1-B2-B2 | Reconcile independently recomputed M19-C2 authorization decisions, fail closed | Accepted |
| M19-D1-B2-B3 | Verify downstream separation from existing single-model B1–B4 disclosure, scoring and evidence workflow | Accepted |

### Implemented D1 production files

- `zenerestimation/comparison/generalization_internal_evidence.py`
- `zenerestimation/comparison/generalization_holdout_boundary.py`
- `zenerestimation/comparison/generalization_authorization_integration.py`

### Relevant D1 tests

- `tests/test_m19_d1_b1_internal_evidence.py`
- `tests/test_m19_d1_b1_real_artifact_integration.py`
- `tests/test_m19_d1_b2_b1_holdout_boundary.py`
- `tests/test_m19_d1_b2_b2_authorization_integration.py`
- `tests/test_m19_d1_b2_b3_downstream_separation.py`

## 2. Test evidence

| Test run | Outcome |
| --- | --- |
| B2-B2 focused acceptance | 12 passed in 1.47 s |
| B1 + B2-B1 + B2-B2 combined | 28 passed in 1.82 s |
| B2-B3 focused acceptance | 10 passed in 1.19 s |
| D1 and existing B3/B4 combined regression | 78 passed in 2.76 s |
| Full repository regression | **3,445 passed, 161 warnings, 1,520.87 s (25:20)** |

Prior full-suite baseline: 3,413 passed, 161 warnings. Current run contains 32 more passing tests; no increase in warning count. Warnings concern existing statsmodels/ARIMA diagnostics and Keras input-shape usage. These warnings are not evidence of failures, but should not be described as resolved.

## 3. Accepted architectural boundaries

1. **Internal selection is not protected-holdout authorization.** Verified M18 internal selection and M17 fold evidence may support an internal selection decision but do not authorize protected-holdout assessment, comparison, ranking, or disclosure.
2. **Internal evidence does not establish external generalization.** D1 reports `holdout_assessed = False` and `generalization_gap = None`.
3. **Authorization is fail-closed.** The current M19-C2 integration does not establish prospective authorization or grant cross-model comparison/ranking. Reconciliation accepts only the expected denied/required/rejected outcomes and rejects forged or inconsistent supplied decisions.
4. **Existing single-model evaluation remains separate.** B1 eligibility, B2 target disclosure, B3 error scoring, and B4 deterministic evidence reconstruction continue to work without new D1 checks inserted into their production APIs.
5. **Digests prove internal consistency, not authenticity.** Trusted origins, independent time provenance, access completeness, and prospective authorization are not established by deterministic digest reconstruction alone.

## 4. Known limitations / non-claims

- The tests establish behavior for examined interfaces; they do **not** prove repository-wide access controls or the absence of every indirect data-access path.
- A complete real M19-C2 `AUTHORIZATION_REQUIRED` reconciliation fixture was not exercised in the focused B2-B2 suite; that path used a controlled stub. A real invalid C2 input path was exercised and rejected.
- Neither independent historical provenance nor a measured external generalization gap is established by D1-B1/B2.
- No protected-holdout permissions are granted by this checkpoint.

## 5. Git working-tree audit (user-provided outputs, 2026-11-20)

**Modified tracked files (5):**

- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT_HISTORY.md`
- `docs/RELEASE_NOTES.md`
- `docs/draft/development_flow.docx`
- `pyproject.toml`

**Tracked diff:** 5 files changed, 107 insertions(+), 3 deletions(-). The Word document is a binary diff. These numbers exclude all untracked files.

**Untracked D1 files:** all three D1 production modules and the five D1 test files above appear in `git status --short` as untracked. Numerous earlier M16–M19 source files, test files, data exports, backups, archives and reports are also untracked. The status is therefore **not a clean or isolated release state**.

**Whitespace/line endings:** `git diff --check` showed LF→CRLF conversion warnings for three documentation files but no whitespace-error lines. The warnings should be managed through repository line-ending policy, not broad automatic rewriting during this checkpoint.

## 6. Recommended local checkpoint procedure

This record is a **documentation checkpoint**, not a Git commit. Do not use `git add .` given the extensive unrelated untracked files.

1. Inspect the exact D1 file list and verify that no dependencies are missing from a prospective commit.
2. Review changes in the tracked documentation and `pyproject.toml` separately; they may contain work from other milestones.
3. Stage only approved paths with explicit `git add -- <path>` arguments when a local commit is desired.
4. Use `git diff --cached --check` and `git diff --cached --stat` before committing.
5. Keep remote synchronization deferred until the broader milestone is ready, consistent with the project workflow.

### Suggested read-only commands

```powershell
git status --short
git diff --check
git diff --stat
git ls-files --others --exclude-standard zenerestimation/comparison tests | Select-String 'generalization_|test_m19_d1_'
```

## 7. Next decision

The D1-B1/B2 technical work is accepted. Before proceeding to the next D1 work package, determine the remaining D1 scope from the authoritative roadmap or milestone plan, rather than inferring the next section from filenames. Maintain a separate local checkpoint record until the repository's large untracked inventory is reconciled.
