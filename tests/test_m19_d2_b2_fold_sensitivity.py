"""M19-D2-B2 tests: deterministic LOFO on M18-verified evidence."""
from __future__ import annotations

from copy import deepcopy
from math import isclose

import pytest

from zenerestimation.assessment.fold_sensitivity import assess_internal_fold_sensitivity
from tests.test_m18_accuracy_profile_ranking import trusted_context, make_result


def sample():
    trusted = trusted_context()
    results = {
        "LSTM": make_result("LSTM", trusted, errors=((0.2, 0.2), (0.6, 0.6))),
        "GRU": make_result("GRU", trusted, errors=((0.1, 0.1), (0.3, 0.3))),
    }
    return results, trusted


def test_correct_lofo_arithmetic_and_fold_identity():
    results, trusted = sample()
    report = assess_internal_fold_sensitivity(results, validation_context=trusted)
    assert report["method"] == "leave_one_fold_out_reaggregation"
    assert report["fold_count"] == 2
    assert [r["model"] for r in report["models"]] == ["GRU", "LSTM"]
    lstm = report["models"][1]
    assert lstm["status"] == "available"
    assert lstm["baseline_rmse"] == pytest.approx(0.4)
    assert [x["lofo_rmse"] for x in lstm["leave_one_fold_out"]] == pytest.approx([0.6, 0.2])
    assert [x["influence"] for x in lstm["leave_one_fold_out"]] == pytest.approx([0.2, -0.2])
    assert lstm["max_absolute_influence"] == pytest.approx(0.2)
    assert lstm["leave_one_fold_out"][0]["validation_dates"] == trusted["fold_validation_dates"][0]


def test_order_invariance_and_determinism():
    results, trusted = sample()
    a = assess_internal_fold_sensitivity(results, validation_context=trusted)
    b = assess_internal_fold_sensitivity(dict(reversed(list(results.items()))), validation_context=trusted)
    assert a == b
    assert a == assess_internal_fold_sensitivity(results, validation_context=trusted)


def test_read_only():
    results, trusted = sample()
    original_results = {k: v.to_dict() for k, v in results.items()}
    original_context = deepcopy(trusted)
    assess_internal_fold_sensitivity(results, validation_context=trusted)
    assert {k: v.to_dict() for k, v in results.items()} == original_results
    assert trusted == original_context


@pytest.mark.parametrize("tamper", ["negative", "nan", "score", "calendar", "alignment", "profile"])
def test_tampering_rejected(tamper):
    results, trusted = sample()
    candidate = results["LSTM"].candidates[0]
    if tamper == "negative":
        candidate.metadata["fold_metrics"][0]["rmse"] = -1.0
    elif tamper == "nan":
        candidate.metadata["fold_metrics"][0]["rmse"] = float("nan")
    elif tamper == "score":
        candidate.metadata["fold_metrics"][0]["rmse"] = 9.0
    elif tamper == "calendar":
        candidate.metadata["fold_alignment"][0]["forecast_dates"][0] = "2030-03-01"
    elif tamper == "alignment":
        candidate.metadata["fold_alignment"][0]["status"] = "positional_only"
    else:
        candidate.metadata["accuracy_profile"]["by_date"][0]["predicted_microVolt"] += 1
    with pytest.raises(ValueError):
        assess_internal_fold_sensitivity(results, validation_context=trusted)


def test_external_scope_rejected():
    results, trusted = sample()
    trusted["scope"] = "external_holdout"
    with pytest.raises(ValueError):
        assess_internal_fold_sensitivity(results, validation_context=trusted)


def test_missing_context_rejected():
    results, _ = sample()
    with pytest.raises(ValueError):
        assess_internal_fold_sensitivity(results, validation_context=None)


def test_single_fold_math_is_unavailable():
    # D2-B1's M18 verifier may reject a single-fold input in a particular
    # fixture; here we assert the D2-B2 policy only when verified evidence
    # actually supplies one fold, without constructing a forged public input.
    from unittest.mock import patch
    results, trusted = sample()
    one = {
        "evidence_version": "m19_d2_verified_internal_folds_v1",
        "scope": "internal_expanding_window",
        "dataset_id": "synthetic",
        "dataset_fingerprint": "sha256:" + "a" * 64,
        "fold_validation_dates": [["2022-03-01"]],
        "models": [{"model": "LSTM", "score": 0.2, "fold_rmse": [0.2]}],
    }
    with patch("zenerestimation.assessment.fold_sensitivity.extract_verified_internal_fold_scores", return_value=one):
        report = assess_internal_fold_sensitivity(results, validation_context=trusted)
    row = report["models"][0]
    assert row["status"] == "unavailable_insufficient_folds"
    assert row["leave_one_fold_out"] == []
    assert row["max_absolute_influence"] is None
