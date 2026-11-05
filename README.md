# Milestone 16.4-C1.1 — Common-interval eligibility correction

Replace the two files at their matching repository-relative paths. The new test file is additive.

The historical evidence builder continues to require exact timestamp alignment and to exclude
missing measured observations. The common-interval comparator now excludes models with no
usable historical fitted observations (such as the current Kalman implementation) from the
date intersection, while recording their exclusion explicitly. Its results are historical
reconstruction diagnostics, not out-of-sample forecast rankings.

The runner prints included/excluded models and attempts to capture ARIMA optimizer convergence
only if an accessible statsmodels result exposes `mle_retvals['converged']`; otherwise it
records `null` rather than claiming convergence. Existing frozen benchmark code is untouched.

From the repository root:

    python -m pytest tests/test_16_4_c1_common_interval_eligibility.py -q
    python -m pytest tests/test_16_4_c1_historical_fit_evidence.py -q
    python examples/diagnostics/run_16_4_c1_historical_fit.py --dry-run
    python examples/diagnostics/run_16_4_c1_historical_fit.py

This patch was tested against the supplied files, not the complete local repository.
