"""M19-C1 focused benchmark ingestion contract (no ranking)."""
from copy import deepcopy
import json

import pytest

from tests.test_m19_b3_b3_holdout_error_scoring import case  # noqa: F401
from tests.test_m19_b3_b4_evaluation_evidence import chain  # noqa: F401
from zenerestimation.assessment.holdout_evaluation_evidence import build_holdout_evaluation_evidence
from zenerestimation.assessment.holdout_target_disclosure import _digest
from zenerestimation.comparison.benchmark_contract import (
    BENCHMARK_ENTRY_VERSION, build_benchmark_entry, verify_benchmark_entry,
)


@pytest.fixture
def inputs(chain):
    evidence = build_holdout_evaluation_evidence(**chain)
    return {**chain, "evaluation_evidence": evidence}


def _build(inputs, run_id="410.GRU.seed0", model_family="GRU"):
    return build_benchmark_entry(run_id=run_id, model_family=model_family, **inputs)


def _verify(entry, inputs):
    return verify_benchmark_entry(entry, **inputs)


def test_valid_chain_ingestion(inputs):
    entry = _build(inputs)
    assert entry["benchmark_entry_version"] == BENCHMARK_ENTRY_VERSION
    assert entry["status"] == "VERIFIED_FOR_BENCHMARK_INGESTION"
    assert entry["metric_source"] == "pooled_holdout"
    assert entry["evaluation_evidence_digest"] == inputs["evaluation_evidence"]["evaluation_evidence_digest"]
    assert entry["metrics"] == inputs["evaluation_evidence"]["metrics"]
    assert entry["by_date"] == inputs["evaluation_evidence"]["by_date"]
    assert entry["scientific_independence"] == "not_established"
    assert entry["authentication"] == "not_established"
    assert entry["benchmark_entry_digest"] == _digest({k: v for k, v in entry.items() if k != "benchmark_entry_digest"})
    assert _verify(entry, inputs)


def test_deterministic_and_json_safe(inputs):
    a, b = _build(inputs), _build(inputs)
    assert a == b
    json.dumps(a, allow_nan=False)


def test_detached_nested_data(inputs):
    a = _build(inputs)
    a["by_date"][0]["signed_error_microVolt"] = 500
    a["metrics"]["rmse_microVolt"] = 500
    a["training_fingerprint"]["digest"] = "changed"
    b = _build(inputs)
    assert b["by_date"][0]["signed_error_microVolt"] != 500
    assert b["metrics"]["rmse_microVolt"] != 500
    assert b["training_fingerprint"]["digest"] != "changed"


@pytest.mark.parametrize("run_id", ["", " bad", "bad id", "a/b", "x" * 129, None, 1, "é", "a\n"])
def test_invalid_run_id(inputs, run_id):
    with pytest.raises(ValueError, match="run_id"):
        _build(inputs, run_id=run_id)


@pytest.mark.parametrize("family", ["", "Unknown", "gru", "LSTM", "ARIMA", "Kalman", "Hybrid", None, 1])
def test_invalid_or_mismatched_family(inputs, family):
    with pytest.raises(ValueError, match="model_family"):
        _build(inputs, model_family=family)


@pytest.mark.parametrize("path,value", [
    (("metrics", "rmse_microVolt"), 999.0),
    (("by_date", 0, "signed_error_microVolt"), 999.0),
    (("forecast_dates", 0), "2024-09-01"),
    (("evaluation_evidence_digest",), "fake"),
    (("scientific_independence",), "established"),
    (("authentication",), "established"),
])
def test_redigested_b4_tamper_rejected(inputs, path, value):
    forged = inputs["evaluation_evidence"]
    target = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path != ("evaluation_evidence_digest",):
        forged["evaluation_evidence_digest"] = _digest(
            {k: v for k, v in forged.items() if k != "evaluation_evidence_digest"}
        )
    with pytest.raises(ValueError, match="B4 evaluation evidence"):
        _build(inputs)


@pytest.mark.parametrize("upstream,key", [
    ("eligibility_snapshot", "predicted_microVolt"),
    ("disclosure_snapshot", "observed_microVolt"),
    ("scoring_snapshot", "metrics"),
])
def test_upstream_tamper_rejected(inputs, upstream, key):
    target = inputs[upstream][key]
    if isinstance(target, list):
        target[0] = 999.0
    else:
        target["rmse_microVolt"] = 999.0
    with pytest.raises(ValueError, match="B4 evaluation evidence"):
        _build(inputs)


@pytest.mark.parametrize("path,value", [
    (("metrics", "rmse_microVolt"), 999.0),
    (("by_date", 0, "observed_microVolt"), 999.0),
    (("forecast_horizon",), 999),
    (("metric_source",), "fold_mean"),
    (("run_id",), "another-run"),
    (("model_family",), "Hybrid"),
    (("scientific_independence",), "established"),
    (("authentication",), "established"),
])
def test_redigested_entry_tamper_rejected(inputs, path, value):
    entry = _build(inputs)
    target = entry
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    entry["benchmark_entry_digest"] = _digest({k: v for k, v in entry.items() if k != "benchmark_entry_digest"})
    # run_id is caller-supplied identity; verification cannot authenticate its origin.
    if path == ("run_id",):
        assert _verify(entry, inputs)
    else:
        assert not _verify(entry, inputs)


def test_extra_field_rejected(inputs):
    entry = _build(inputs)
    entry["extra"] = True
    assert not _verify(entry, inputs)


def test_missing_field_rejected(inputs):
    entry = _build(inputs)
    del entry["run_id"]
    assert not _verify(entry, inputs)


def test_non_mapping_rejected(inputs):
    assert not _verify(None, inputs)


def test_nonfinite_entry_rejected(inputs):
    entry = _build(inputs)
    entry["metrics"]["rmse_microVolt"] = float("nan")
    assert not _verify(entry, inputs)


def test_different_run_ids_distinct_digests(inputs):
    a = _build(inputs, run_id="seed0")
    b = _build(inputs, run_id="seed1")
    assert a["benchmark_entry_digest"] != b["benchmark_entry_digest"]
    assert a["evaluation_evidence_digest"] == b["evaluation_evidence_digest"]
