"""M19-C2-B2-B2-A: prospective commitments do not authorize holdout use."""
from copy import deepcopy

import pytest

from zenerestimation.comparison.benchmark_authorization import build_benchmark_scope_declaration
from zenerestimation.comparison.benchmark_authorization_prospective import (
    build_prospective_benchmark_commitment,
    verify_prospective_benchmark_commitment,
    assess_prospective_commitment_binding,
)


def _participants():
    return [
        {"run_id": "run.gru", "assessment_id": "assess.gru", "model_family": "GRU"},
        {"run_id": "run.lstm", "assessment_id": "assess.lstm", "model_family": "LSTM"},
    ]


def _commitment():
    return build_prospective_benchmark_commitment(
        benchmark_id="benchmark.410", dataset_id="410",
        forecast_dates=["2025-03-01", "2025-06-01"], participants=_participants(),
    )


def _scope():
    participants = [
        {**p, "benchmark_entry_digest": "sha256:" + char * 64}
        for p, char in zip(_participants(), "ab")
    ]
    return build_benchmark_scope_declaration(
        benchmark_id="benchmark.410", dataset_id="410",
        forecast_dates=["2025-03-01", "2025-06-01"], participants=participants,
    )


def test_matching_commitment_never_authorizes():
    result = assess_prospective_commitment_binding(commitment=_commitment(), scope_declaration=_scope())
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result["commitment_binding_verified"] is True
    assert result["prospective_authorization_verified"] is False
    assert result["comparison_permitted"] is False and result["ranking_permitted"] is False


def test_commitment_does_not_contain_future_entry_digests():
    commitment = _commitment()
    assert verify_prospective_benchmark_commitment(commitment)
    assert all("benchmark_entry_digest" not in p for p in commitment["participants"])
    assert "scope_digest" not in commitment
    assert "issued_at" not in commitment
    assert "approved" not in commitment


def test_retrospective_digest_changes_do_not_change_prospective_commitment():
    scope = _scope()
    scope["participants"][0]["benchmark_entry_digest"] = "sha256:" + "c" * 64
    scope = build_benchmark_scope_declaration(
        benchmark_id=scope["benchmark_id"], dataset_id=scope["dataset_id"],
        forecast_dates=scope["forecast_dates"], participants=scope["participants"],
    )
    result = assess_prospective_commitment_binding(commitment=_commitment(), scope_declaration=scope)
    assert result["commitment_binding_verified"] is True
    assert result["status"] == "AUTHORIZATION_REQUIRED"


@pytest.mark.parametrize("field,changed,reason", [
    ("benchmark_id", "benchmark.110", "BENCHMARK_MISMATCH"),
    ("dataset_id", "110", "DATASET_MISMATCH"),
    ("forecast_dates", ["2025-09-01", "2025-12-01"], "TIMESTAMP_MISMATCH"),
])
def test_retrospective_scope_mismatch(field, changed, reason):
    scope = _scope()
    args = {k: scope[k] for k in ("benchmark_id", "dataset_id", "forecast_dates", "participants")}
    args[field] = changed
    changed_scope = build_benchmark_scope_declaration(**args)
    result = assess_prospective_commitment_binding(commitment=_commitment(), scope_declaration=changed_scope)
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert reason in result["reasons"]
    assert not result["comparison_permitted"]


def test_participant_substitution_rejected():
    scope = _scope()
    participants = deepcopy(scope["participants"])
    participants[0]["assessment_id"] = "assess.other"
    changed = build_benchmark_scope_declaration(
        benchmark_id="benchmark.410", dataset_id="410",
        forecast_dates=scope["forecast_dates"], participants=participants,
    )
    result = assess_prospective_commitment_binding(commitment=_commitment(), scope_declaration=changed)
    assert result["reasons"] == ["PARTICIPANT_MISMATCH"]
    assert result["status"] == "AUTHORIZATION_REJECTED"


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(commitment_version="future"),
    lambda c: c.update(authorized_operation="compare"),
    lambda c: c.update(approved=True),
    lambda c: c.update(issued_at="2024-01-01T00:00:00Z"),
    lambda c: c.update(forecast_horizon=True),
    lambda c: c["participants"][0].update(benchmark_entry_digest="sha256:" + "a" * 64),
    lambda c: c["participants"][0].update(run_id="run.lstm"),
    lambda c: c.update(forecast_dates=["2025-06-01", "2025-03-01"]),
    lambda c: c.update(commitment_digest="sha256:" + "f" * 64),
])
def test_invalid_or_tampered_commitment_rejected(mutation):
    commitment = _commitment()
    mutation(commitment)
    assert not verify_prospective_benchmark_commitment(commitment)
    with pytest.raises(ValueError):
        assess_prospective_commitment_binding(commitment=commitment, scope_declaration=_scope())


def test_duplicate_assessment_ids_rejected():
    participants = _participants()
    participants[1]["assessment_id"] = participants[0]["assessment_id"]
    with pytest.raises(ValueError):
        build_prospective_benchmark_commitment(
            benchmark_id="benchmark.410", dataset_id="410",
            forecast_dates=["2025-03-01", "2025-06-01"], participants=participants,
        )


def test_deterministic_and_read_only():
    commitment, scope = _commitment(), _scope()
    old_commitment, old_scope = deepcopy(commitment), deepcopy(scope)
    first = assess_prospective_commitment_binding(commitment=commitment, scope_declaration=scope)
    second = assess_prospective_commitment_binding(commitment=commitment, scope_declaration=scope)
    assert first == second
    assert first["decision_digest"].startswith("sha256:")
    assert commitment == old_commitment and scope == old_scope
