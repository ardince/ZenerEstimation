"""M19-C2-B1 isolated scope contract tests; genuine C1/M18 integration follows in B3."""
from copy import deepcopy
from types import SimpleNamespace
import pytest

import zenerestimation.comparison.benchmark_authorization as auth

D = ["2024-03-01", "2024-06-01"]


def _participants():
    return [
        dict(run_id="run.GRU", assessment_id="assess.GRU", model_family="GRU", benchmark_entry_digest="sha256:" + "a" * 64),
        dict(run_id="run.LSTM", assessment_id="assess.LSTM", model_family="LSTM", benchmark_entry_digest="sha256:" + "b" * 64),
    ]


def _scope(**kwargs):
    args = dict(benchmark_id="bench.410", dataset_id="410", forecast_dates=D, participants=_participants())
    args.update(kwargs)
    return auth.build_benchmark_scope_declaration(**args)


def _inputs():
    return [SimpleNamespace(entry=p) for p in _participants()]


@pytest.fixture
def aligned(monkeypatch):
    # Isolate B1 scope logic. Real C1/C2-A verification is exercised in B3.
    monkeypatch.setattr(auth, "assess_scientific_alignment", lambda inputs: dict(
        status="SCIENTIFICALLY_ALIGNED", shared_dataset_id="410", shared_horizon=2,
        shared_forecast_dates=D, observation_digest="c" * 64,
    ))


def test_build_verify_deterministic_detached():
    participants = _participants()
    first = _scope()
    assert auth.verify_benchmark_scope_declaration(first)
    assert first == _scope(participants=list(reversed(participants)))
    participants[0]["run_id"] = "mutated"
    assert auth.verify_benchmark_scope_declaration(first)
    assert first["scope_status"] == "SELF_DECLARED_SCOPE_ONLY"
    assert "authorization_issued_at" not in first


def test_valid_scope_is_not_authorization(aligned):
    result = auth.assess_benchmark_scope(inputs=_inputs(), declaration=_scope())
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result["reasons"] == []
    assert result["authorization"] == "not_established"
    assert result["authentication"] == "not_established"
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False
    assert result["decision_digest"].startswith("sha256:")
    assert len(result["decision_digest"]) == 71
    assert result == auth.assess_benchmark_scope(inputs=_inputs()[::-1], declaration=_scope())


@pytest.mark.parametrize("change,reason", [
    ({"dataset_id": "110"}, "DATASET_SCOPE_MISMATCH"),
    ({"forecast_dates": ["2024-06-01", "2024-09-01"]}, "TIMESTAMP_SCOPE_MISMATCH"),
])
def test_scope_mismatch_rejected(aligned, change, reason):
    result = auth.assess_benchmark_scope(inputs=_inputs(), declaration=_scope(**change))
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert reason in result["reasons"]
    assert not result["comparison_permitted"]


def test_participant_substitution_rejected(aligned):
    altered = _participants()
    altered[0]["benchmark_entry_digest"] = "sha256:" + "f" * 64
    result = auth.assess_benchmark_scope(inputs=_inputs(), declaration=_scope(participants=altered))
    assert result["reasons"] == ["PARTICIPANT_SCOPE_MISMATCH"]


def test_aligned_required_even_with_matching_scope(monkeypatch):
    monkeypatch.setattr(auth, "assess_scientific_alignment", lambda inputs: dict(
        status="INCOMPARABLE", shared_dataset_id=None, shared_horizon=None,
        shared_forecast_dates=None, observation_digest=None,
    ))
    result = auth.assess_benchmark_scope(inputs=_inputs(), declaration=_scope())
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert "SCIENTIFIC_ALIGNMENT_FAILED" in result["reasons"]


@pytest.mark.parametrize("field,value", [
    ("scope_status", "AUTHORIZED"), ("purpose", "ranking"),
    ("benchmark_id", ""), ("scope_version", "future"),
    ("forecast_horizon", True), ("scope_digest", "0" * 64),
])
def test_tampered_declaration_rejected(field, value):
    item = _scope()
    item[field] = value
    assert not auth.verify_benchmark_scope_declaration(item)


def test_redigested_false_approval_still_rejected():
    item = _scope()
    item["scope_status"] = "AUTHORIZED"
    item["scope_digest"] = auth._digest({k: v for k, v in item.items() if k != "scope_digest"})
    assert not auth.verify_benchmark_scope_declaration(item)


@pytest.mark.parametrize("participants", [
    [], _participants()[:1],
    [dict(_participants()[0]), dict(_participants()[0], run_id="another")],
    [dict(_participants()[0]), dict(_participants()[1], run_id="run.GRU")],
    [dict(_participants()[0]), dict(_participants()[1], benchmark_entry_digest="xyz")],
])
def test_invalid_participant_sets(participants):
    with pytest.raises(ValueError):
        _scope(participants=participants)


def test_unknown_fields_fail_closed():
    item = _scope()
    item["independent_approval"] = True
    assert not auth.verify_benchmark_scope_declaration(item)


def test_input_verification_failure_propagates(monkeypatch):
    def fail(_):
        raise ValueError("C1 upstream verification failed")
    monkeypatch.setattr(auth, "assess_scientific_alignment", fail)
    with pytest.raises(ValueError, match="C1 upstream"):
        auth.assess_benchmark_scope(inputs=_inputs(), declaration=_scope())


@pytest.mark.parametrize("invalid_digest", [
    "a" * 64,
    "SHA256:" + "a" * 64,
    "sha256:" + "A" * 64,
    "sha256:" + "a" * 63,
    "sha256:" + "g" * 64,
])
def test_participant_digest_requires_c1_format(invalid_digest):
    participants = _participants()
    participants[0]["benchmark_entry_digest"] = invalid_digest
    with pytest.raises(ValueError, match="benchmark_entry_digest"):
        _scope(participants=participants)
