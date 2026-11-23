"""M19-C2-A genuine M18 -> B1 -> B2 -> B3 -> B4 -> C1 -> C2-A tests.

The accepted M18 ProbeModel is registered as GRU. Repeated C1 entries
represent the SAME approved execution, not independent model assessments.
Scientific alignment is informational and does not authorize comparison.
"""
from copy import deepcopy
import json

import pytest

from tests.test_m19_c1_real_m18_integration import (
    genuine_chain,  # noqa: F401 -- pytest fixture: genuine M18 execution
    _copy_evidence_inputs,
)
from zenerestimation.assessment.holdout_target_disclosure import _digest
from zenerestimation.comparison.benchmark_contract import (
    build_benchmark_entry,
    verify_benchmark_entry,
)
from zenerestimation.comparison.benchmark_comparability import (
    VerifiedBenchmarkInput,
    assess_scientific_alignment,
)
from tests.test_m18_forecast_execution_contract import ProbeModel
from tests.test_m18_forecast_plan_contract import training as processed_training
from tests.test_m19_b3_b1_m18_real_integration import declaration, full_frame


def _member(inputs, run_id):
    entry = build_benchmark_entry(run_id=run_id, model_family="GRU", **inputs)
    assert verify_benchmark_entry(entry, **inputs)
    return VerifiedBenchmarkInput(entry=entry, **inputs)


def _pair(inputs):
    # Two run identifiers refer to one verified M18 chain, NOT two
    # independently authorized model evaluations.
    return [
        _member(inputs, "410.GRU.real-m18.c2a-a"),
        _member(inputs, "410.GRU.real-m18.c2a-b"),
    ]


def _assert_no_new_execution(events):
    assert ProbeModel.events == events


def test_genuine_m18_chain_reaches_c2a_without_new_model_execution(genuine_chain):
    inputs, events = genuine_chain
    pair = _pair(inputs)
    result = assess_scientific_alignment(pair)
    assert result["contract_version"] == "m19_c2a_scientific_alignment_v1"
    assert result["status"] == "SCIENTIFICALLY_ALIGNED"
    assert result["reasons"] == []
    assert result["shared_dataset_id"] == inputs["declaration"].dataset_id
    assert result["shared_forecast_dates"] == list(inputs["declaration"].forecast_dates)
    assert result["shared_horizon"] == len(inputs["declaration"].forecast_dates)
    assert result["observation_digest"].startswith("sha256:")
    _assert_no_new_execution(events)


def test_real_alignment_never_grants_authorization_or_ranking(genuine_chain):
    inputs, events = genuine_chain
    result = assess_scientific_alignment(_pair(inputs))
    assert result["authorization"] == "not_established"
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False
    _assert_no_new_execution(events)


def test_real_alignment_deterministic_order_independent_and_read_only(genuine_chain):
    inputs, events = genuine_chain
    pair = _pair(inputs)
    before_entries = [deepcopy(member.entry) for member in pair]
    before_snapshots = {
        key: deepcopy(value) for key, value in inputs.items() if key != "declaration"
    }
    first = assess_scientific_alignment(pair)
    assert first == assess_scientific_alignment(pair)
    assert first == assess_scientific_alignment(pair[::-1])
    json.dumps(first, allow_nan=False)
    assert [member.entry for member in pair] == before_entries
    assert all(inputs[key] == value for key, value in before_snapshots.items())
    _assert_no_new_execution(events)


def test_real_observation_digest_is_from_verified_observations(genuine_chain):
    inputs, events = genuine_chain
    pair = _pair(inputs)
    result = assess_scientific_alignment(pair)
    entry = pair[0].entry
    expected = _digest({
        "dates": entry["forecast_dates"],
        "observed_microVolt": [row["observed_microVolt"] for row in entry["by_date"]],
    })
    assert result["observation_digest"] == expected
    _assert_no_new_execution(events)


@pytest.mark.parametrize("path,value", [
    (("forecast_dates", 0), "2024-09-01"),
    (("by_date", 0, "observed_microVolt"), 999.0),
    (("metrics", "rmse_microVolt"), 999.0),
    (("authentication",), "established"),
])
def test_real_entry_tampering_fails_c1_before_alignment(genuine_chain, path, value):
    inputs, events = genuine_chain
    pair = _pair(inputs)
    node = pair[1].entry
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match="C1 upstream verification"):
        assess_scientific_alignment(pair)
    _assert_no_new_execution(events)


def test_real_cross_run_upstream_substitution_fails_before_alignment(genuine_chain):
    inputs, events = genuine_chain
    pair = _pair(inputs)
    substituted = _copy_evidence_inputs(inputs)
    substituted["evaluation_evidence"]["metrics"]["rmse_microVolt"] = 999.0
    # Even if a forged B4 evidence digest is recalculated, the full
    # independent B1-B4 reconstruction must reject the substitution.
    evidence = substituted["evaluation_evidence"]
    evidence["evaluation_evidence_digest"] = _digest({
        k: v for k, v in evidence.items() if k != "evaluation_evidence_digest"
    })
    pair[1] = VerifiedBenchmarkInput(entry=pair[1].entry, **substituted)
    with pytest.raises(ValueError, match="C1 upstream verification"):
        assess_scientific_alignment(pair)
    _assert_no_new_execution(events)


def test_real_duplicate_run_identifier_rejected(genuine_chain):
    inputs, events = genuine_chain
    member = _member(inputs, "410.GRU.real-m18.duplicate")
    with pytest.raises(ValueError, match="duplicate run_id"):
        assess_scientific_alignment([member, member])
    _assert_no_new_execution(events)
