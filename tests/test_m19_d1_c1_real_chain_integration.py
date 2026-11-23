"""M19-D1-C1 integration through real D1-B2-B2 and internal verification.

Only the C2 decision is stubbed for the authorization-required branch;
the rejected branch invokes the real C2 verifier. No protected targets are read.
"""
from copy import deepcopy

import pytest

from tests.test_m19_d1_b2_b2_authorization_integration import (
    _internal, _c2_inputs, _decision,
)
from zenerestimation.comparison.generalization_gap_assessment import (
    VERSION, assess_generalization_gap,
)


def _run(**overrides):
    args = {**_internal(), "benchmark_authorization_inputs": _c2_inputs()}
    args.update(overrides)
    return assess_generalization_gap(**args)


def _assert_denied(result, status):
    assert result["assessment_version"] == VERSION
    assert result["status"] == status
    assert result["internal_evidence_verified"] is True
    assert result["selected_model"] == "GRU"
    assert result["comparability_verified"] is False
    assert result["holdout_assessed"] is False
    assert result["generalization_gap"] is None
    assert result["generalization_gap_metric"] is None
    for field in (
        "protected_holdout_assessment_permitted",
        "protected_holdout_comparison_permitted",
        "protected_holdout_ranking_permitted",
        "protected_holdout_disclosure_permitted",
    ):
        assert result[field] is False


def test_real_c2_rejection_propagates_through_full_d1_chain():
    result = _run()
    _assert_denied(result, "AUTHORIZATION_REJECTED")
    assert result["benchmark_authorization_status"] == "AUTHORIZATION_REJECTED"
    assert result["benchmark_authorization_decision_digest"].startswith("sha256:")


def test_real_d1_chain_with_controlled_insufficient_c2(monkeypatch):
    # Only C2 is controlled: D1-C1, D1-B2-B2, and internal verification are real.
    decision = _decision("AUTHORIZATION_REQUIRED")
    monkeypatch.setattr(
        "zenerestimation.comparison.generalization_authorization_integration."
        "assess_independent_benchmark_authorization",
        lambda **kwargs: deepcopy(decision),
    )
    result = _run(supplied_authorization_decision=deepcopy(decision))
    _assert_denied(result, "AUTHORIZATION_REQUIRED")
    assert result["benchmark_authorization_status"] == "AUTHORIZATION_REQUIRED"
    assert result["benchmark_authorization_decision_digest"] == decision["decision_digest"]
    assert result["comparability_verified"] is False


def test_supplied_decision_mismatch_rejected_with_reduced_record(monkeypatch):
    decision = _decision()
    monkeypatch.setattr(
        "zenerestimation.comparison.generalization_authorization_integration."
        "assess_independent_benchmark_authorization",
        lambda **kwargs: deepcopy(decision),
    )
    forged = deepcopy(decision)
    forged["comparison_permitted"] = True
    result = _run(supplied_authorization_decision=forged)
    _assert_denied(result, "AUTHORIZATION_REJECTED")
    assert result["reasons"] == ["SUPPLIED_AUTHORIZATION_DECISION_MISMATCH"]
    assert result["benchmark_authorization_decision_digest"] is None
    assert result["benchmark_authorization_status"] is None


def test_tampered_internal_selection_rejected_before_c2(monkeypatch):
    def forbidden_c2(**kwargs):
        pytest.fail("C2 must not execute after invalid internal selection")
    monkeypatch.setattr(
        "zenerestimation.comparison.generalization_authorization_integration."
        "assess_independent_benchmark_authorization", forbidden_c2,
    )
    internal = _internal()
    internal["selection_artifact"]["selection_decision_record"]["decision"]["selected_model"] = "LSTM"
    with pytest.raises((TypeError, ValueError)):
        _run(**internal)


def test_real_chain_deterministic_and_read_only():
    internal = _internal()
    c2 = _c2_inputs()
    before_internal, before_c2 = deepcopy(internal), deepcopy(c2)
    first = _run(**internal, benchmark_authorization_inputs=c2)
    second = _run(**internal, benchmark_authorization_inputs=c2)
    assert first == second
    assert internal == before_internal
    assert c2 == before_c2
    first["reasons"].append("local change")
    assert second["reasons"] != first["reasons"]
