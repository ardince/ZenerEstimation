"""M19-C1 genuine M18 -> B1 -> B2 -> B3 -> B4 -> C1 integration.

The accepted M18 ProbeModel is registered as GRU in the test registry.
This verifies contract wiring, not performance of a trained GRU or external
scientific independence/authentication.
"""
from copy import deepcopy
import json

import pytest

from tests.test_m18_forecast_plan_contract import training as processed_training
from tests.test_m18_forecast_execution_contract import ProbeModel
from tests.test_m19_b3_b1_m18_real_integration import declaration, full_frame, _reset_probe
from tests.test_m19_b3_b4_m18_real_integration import _complete
from zenerestimation.assessment.holdout_target_disclosure import _digest
from zenerestimation.comparison.benchmark_contract import (
    build_benchmark_entry, verify_benchmark_entry,
)


def _inputs(processed_training, full_frame, declaration):
    eligibility, disclosure, scoring, evidence = _complete(
        processed_training, full_frame, declaration,
    )
    return dict(
        declaration=declaration,
        eligibility_snapshot=eligibility,
        disclosure_snapshot=disclosure,
        scoring_snapshot=scoring,
        evaluation_evidence=evidence,
    )


def _copy_evidence_inputs(inputs):
    """Keep the immutable declaration; detach only JSON evidence snapshots.

    AssessmentDeclaration contains a mappingproxy and is not deepcopy-able.
    """
    return {
        key: value if key == "declaration" else deepcopy(value)
        for key, value in inputs.items()
    }


def _build(inputs):
    return build_benchmark_entry(run_id="410.GRU.real-m18", model_family="GRU", **inputs)


def _redigest(record, field):
    record[field] = _digest({k: v for k, v in record.items() if k != field})


@pytest.fixture
def genuine_chain(processed_training, full_frame, declaration):
    _reset_probe()
    inputs = _inputs(processed_training, full_frame, declaration)
    events = deepcopy(ProbeModel.events)
    assert [event[0] for event in events] == ["construct", "fit", "predict"]
    assert inputs["evaluation_evidence"]["selected_model"] == "GRU"
    return inputs, events


def test_real_m18_chain_ingested_and_verified(genuine_chain):
    inputs, events = genuine_chain
    entry = _build(inputs)
    assert entry["status"] == "VERIFIED_FOR_BENCHMARK_INGESTION"
    assert entry["model_family"] == entry["selected_model"] == "GRU"
    assert entry["forecast_dates"] == list(inputs["declaration"].forecast_dates)
    assert entry["metric_source"] == "pooled_holdout"
    assert entry["metrics"] == inputs["scoring_snapshot"]["metrics"]
    assert entry["by_date"] == inputs["scoring_snapshot"]["by_date"]
    assert entry["evaluation_evidence_digest"] == inputs["evaluation_evidence"]["evaluation_evidence_digest"]
    assert entry["scientific_independence"] == "not_established"
    assert entry["authentication"] == "not_established"
    assert verify_benchmark_entry(entry, **inputs)
    assert ProbeModel.events == events


def test_real_chain_deterministic_detached_and_no_execution(genuine_chain):
    inputs, events = genuine_chain
    first = _build(inputs)
    assert first == _build(inputs)
    json.dumps(first, allow_nan=False)
    first["by_date"][0]["observed_microVolt"] = 999.0
    first["metrics"]["rmse_microVolt"] = 999.0
    assert not verify_benchmark_entry(first, **inputs)
    assert _build(inputs)["by_date"] == inputs["evaluation_evidence"]["by_date"]
    assert ProbeModel.events == events


@pytest.mark.parametrize("field,value", [
    ("selected_model", "LSTM"),
    ("decision_digest", "sha256:other"),
    ("execution_artifact_digest", "sha256:other"),
    ("predicted_microVolt", [999.0, 999.0]),
])
def test_real_b1_substitution_fails_closed(genuine_chain, field, value):
    inputs, events = genuine_chain
    forged = _copy_evidence_inputs(inputs)
    forged["eligibility_snapshot"][field] = value
    _redigest(forged["eligibility_snapshot"], "eligibility_digest")
    with pytest.raises(ValueError, match="B4 evaluation evidence"):
        _build(forged)
    assert not verify_benchmark_entry(_build(inputs), **forged)
    assert ProbeModel.events == events


@pytest.mark.parametrize("upstream,field,value,digest", [
    ("disclosure_snapshot", "observed_microVolt", [999.0, 999.0], "disclosure_digest"),
    ("scoring_snapshot", "metrics", {"rmse_microVolt": 999.0}, "scoring_digest"),
])
def test_real_disclosure_or_scoring_substitution_fails_closed(
    genuine_chain, upstream, field, value, digest,
):
    inputs, events = genuine_chain
    forged = _copy_evidence_inputs(inputs)
    forged[upstream][field] = value
    _redigest(forged[upstream], digest)
    with pytest.raises(ValueError, match="B4 evaluation evidence"):
        _build(forged)
    assert not verify_benchmark_entry(_build(inputs), **forged)
    assert ProbeModel.events == events


@pytest.mark.parametrize("path,value", [
    (("metrics", "rmse_microVolt"), 999.0),
    (("by_date", 0, "observed_microVolt"), 999.0),
    (("authentication",), "established"),
])
def test_real_redigested_b4_substitution_fails_closed(genuine_chain, path, value):
    inputs, events = genuine_chain
    forged = _copy_evidence_inputs(inputs)
    node = forged["evaluation_evidence"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    _redigest(forged["evaluation_evidence"], "evaluation_evidence_digest")
    with pytest.raises(ValueError, match="B4 evaluation evidence"):
        _build(forged)
    assert not verify_benchmark_entry(_build(inputs), **forged)
    assert ProbeModel.events == events


def test_real_model_family_misclassification_fails_closed(genuine_chain):
    inputs, events = genuine_chain
    with pytest.raises(ValueError, match="model_family"):
        build_benchmark_entry(run_id="410.GRU.real-m18", model_family="LSTM", **inputs)
    assert ProbeModel.events == events


def test_real_cross_run_evidence_substitution_fails_closed(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    original = _inputs(processed_training, full_frame, declaration)
    changed = full_frame.copy(deep=True)
    changed.loc[changed["ds"] > declaration.development_cutoff, "microVolt"] = [31.9, 32.506]
    _reset_probe()
    alternate = _inputs(processed_training, changed, declaration)
    events = deepcopy(ProbeModel.events)
    assert original["eligibility_snapshot"] == alternate["eligibility_snapshot"]
    assert original["evaluation_evidence"] != alternate["evaluation_evidence"]
    entry = _build(original)
    assert verify_benchmark_entry(entry, **original)
    assert not verify_benchmark_entry(entry, **alternate)
    mixed = {**original, "disclosure_snapshot": alternate["disclosure_snapshot"]}
    with pytest.raises(ValueError, match="B4 evaluation evidence"):
        _build(mixed)
    assert ProbeModel.events == events
