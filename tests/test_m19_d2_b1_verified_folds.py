"""M19-D2-B1: verified fold evidence reuses the M18 ranking boundary."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import importlib.util

import pytest

from zenerestimation.optimization.internal_ranking import (
    extract_verified_internal_fold_scores,
    rank_internal_models,
)


def _fixture_module():
    # Reuse canonical M18-B4 measured-profile fixtures without modifying them.
    path = Path(__file__).with_name("test_m18_accuracy_profile_ranking.py")
    spec = importlib.util.spec_from_file_location("m18_b4_d2_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _valid():
    f = _fixture_module()
    trusted = f.trusted_context()
    results = {
        "LSTM": f.make_result("LSTM", trusted, errors=((0.3, 0.3), (0.5, 0.5))),
        "GRU": f.make_result("GRU", trusted, errors=((0.1, 0.1), (0.3, 0.3))),
    }
    return trusted, results


def test_extracts_verified_fold_scores_and_preserves_ranking():
    trusted, results = _valid()
    before = rank_internal_models(results, validation_context=trusted)
    evidence = extract_verified_internal_fold_scores(results, validation_context=trusted)
    assert evidence["scope"] == "internal_expanding_window"
    assert evidence["dataset_id"] == trusted["dataset_id"]
    assert evidence["dataset_fingerprint"] == trusted["dataset_fingerprint"]
    assert evidence["fold_validation_dates"] == trusted["fold_validation_dates"]
    assert [row["model"] for row in evidence["models"]] == ["GRU", "LSTM"]
    assert evidence["models"][0]["fold_rmse"] == pytest.approx([0.1, 0.3])
    assert evidence["models"][1]["fold_rmse"] == pytest.approx([0.3, 0.5])
    for row in evidence["models"]:
        assert row["score"] == pytest.approx(sum(row["fold_rmse"]) / len(row["fold_rmse"]))
    assert rank_internal_models(results, validation_context=trusted) == before
    assert set(before) == {"ranking", "ineligible", "selected_model", "ranking_version"}


def test_deterministic_order_and_read_only():
    trusted, results = _valid()
    original_context = deepcopy(trusted)
    original_results = {k: v.to_dict() for k, v in results.items()}
    a = extract_verified_internal_fold_scores(results, validation_context=trusted)
    b = extract_verified_internal_fold_scores(dict(reversed(list(results.items()))), validation_context=trusted)
    assert a == b
    a["models"][0]["fold_rmse"][0] = 999.0
    assert extract_verified_internal_fold_scores(results, validation_context=trusted) == b
    assert trusted == original_context
    assert {k: v.to_dict() for k, v in results.items()} == original_results


@pytest.mark.parametrize("mutation", [
    "fold_rmse", "alignment", "profile", "provenance", "aggregate", "scope"
])
def test_rejects_tampered_evidence(mutation):
    trusted, results = _valid()
    selected = results["LSTM"].candidates[0]
    if mutation == "fold_rmse":
        selected.metadata["fold_metrics"][0]["rmse"] += 1.0
    elif mutation == "alignment":
        selected.metadata["fold_alignment"][0]["timestamp_match"] = False
    elif mutation == "profile":
        selected.metadata["accuracy_profile"]["by_date"][0]["predicted_microVolt"] += 1.0
    elif mutation == "provenance":
        selected.metadata["internal_provenance"]["dataset_id"] = "other"
    elif mutation == "aggregate":
        selected.metrics["rmse"] += 1.0
    else:
        trusted["scope"] = "external_holdout"
    with pytest.raises(ValueError):
        extract_verified_internal_fold_scores(results, validation_context=trusted)


def test_rejects_missing_context():
    _, results = _valid()
    with pytest.raises((TypeError, ValueError)):
        extract_verified_internal_fold_scores(results, validation_context=None)
