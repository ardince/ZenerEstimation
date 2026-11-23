"""M19-D1-C1 isolated facade contract tests.

Run in the ZenerEstimation repository with pytest. These tests stub only
the already-tested D1-B2-B2 reconciliation boundary.
"""
from copy import deepcopy

import pytest

from zenerestimation.comparison import generalization_gap_assessment as m


def _upstream(status="AUTHORIZATION_REQUIRED"):
    return {
        "status": status,
        "reasons": ["PROTECTED_HOLDOUT_AUTHORIZATION_NOT_ESTABLISHED"]
        if status == "AUTHORIZATION_REQUIRED"
        else ["INDEPENDENT_BENCHMARK_AUTHORIZATION_REJECTED"],
        "internal_evidence_verified": True,
        "selected_model": "GRU",
        "selection_decision_digest": "sha256:selection",
        "benchmark_scope_digest": "sha256:scope",
        "benchmark_authorization_decision_digest": "sha256:decision",
        "benchmark_authorization_status": status,
        "holdout_assessed": False,
        "generalization_gap": None,
        **{field: False for field in m._DENIED_FIELDS},
    }


def _invoke():
    return m.assess_generalization_gap(
        selection_artifact={}, results={}, validation_context={},
        benchmark_authorization_inputs={},
    )


@pytest.mark.parametrize("status", ["AUTHORIZATION_REQUIRED", "AUTHORIZATION_REJECTED"])
def test_fail_closed_status_propagation(monkeypatch, status):
    monkeypatch.setattr(
        m, "reconcile_generalization_authorization",
        lambda **kwargs: _upstream(status),
    )
    result = _invoke()
    assert result["status"] == status
    assert result["generalization_gap"] is None
    assert result["generalization_gap_metric"] is None
    assert result["comparability_verified"] is False
    assert result["holdout_assessed"] is False
    assert all(result[field] is False for field in m._DENIED_FIELDS)


def test_supplied_decision_mismatch_reduced_rejection(monkeypatch):
    rejected = _upstream("AUTHORIZATION_REJECTED")
    rejected["reasons"] = ["SUPPLIED_AUTHORIZATION_DECISION_MISMATCH"]
    for key in (
        "benchmark_scope_digest",
        "benchmark_authorization_decision_digest",
        "benchmark_authorization_status",
    ):
        rejected.pop(key)
    monkeypatch.setattr(m, "reconcile_generalization_authorization", lambda **kwargs: rejected)
    result = _invoke()
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert result["reasons"] == ["SUPPLIED_AUTHORIZATION_DECISION_MISMATCH"]
    assert result["benchmark_scope_digest"] is None
    assert result["benchmark_authorization_decision_digest"] is None


@pytest.mark.parametrize("field,value", [
    ("status", "AUTHORIZED"),
    ("internal_evidence_verified", False),
    ("holdout_assessed", True),
    ("generalization_gap", 0.2),
    ("protected_holdout_assessment_permitted", True),
    ("protected_holdout_comparison_permitted", True),
    ("protected_holdout_ranking_permitted", True),
    ("protected_holdout_disclosure_permitted", True),
    ("selected_model", None),
    ("selection_decision_digest", None),
    ("reasons", []),
])
def test_contradictory_upstream_is_rejected(monkeypatch, field, value):
    bad = _upstream()
    bad[field] = value
    monkeypatch.setattr(m, "reconcile_generalization_authorization", lambda **kwargs: bad)
    with pytest.raises(ValueError):
        _invoke()


def test_upstream_exception_propagates(monkeypatch):
    def reject(**kwargs):
        raise ValueError("invalid internal evidence")
    monkeypatch.setattr(m, "reconcile_generalization_authorization", reject)
    with pytest.raises(ValueError, match="invalid internal evidence"):
        _invoke()


def test_repeated_read_only_detached(monkeypatch):
    upstream = _upstream()
    original = deepcopy(upstream)
    monkeypatch.setattr(m, "reconcile_generalization_authorization", lambda **kwargs: upstream)
    first, second = _invoke(), _invoke()
    assert first == second
    assert upstream == original
    first["reasons"].append("local mutation")
    assert upstream == original
    assert second["reasons"] == original["reasons"]


def test_no_comparability_claim_from_matching_model_name(monkeypatch):
    upstream = _upstream()
    upstream["selected_model"] = "GRU"
    monkeypatch.setattr(m, "reconcile_generalization_authorization", lambda **kwargs: upstream)
    assert _invoke()["comparability_verified"] is False
