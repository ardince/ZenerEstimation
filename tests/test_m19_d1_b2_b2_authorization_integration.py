
"""M19-D1-B2-B2: fail-closed authorization reconciliation tests."""

from copy import deepcopy

import pytest

from tests.test_m18_selection_audit_contract import (
    context,
    unique_results,
)
from zenerestimation.optimization.internal_selection_artifact import (
    build_internal_selection_artifact,
)
from zenerestimation.assessment.holdout_target_disclosure import _digest
from zenerestimation.comparison.benchmark_authorization_integration import (
    ASSESSMENT_VERSION as C2_VERSION,
)
from zenerestimation.comparison.generalization_authorization_integration import (
    reconcile_generalization_authorization,
)


def _internal():
    results = unique_results()
    trusted = context()
    artifact = build_internal_selection_artifact(
        results,
        validation_context=trusted,
        authorized=True,
    )
    return {
        "selection_artifact": artifact,
        "results": results,
        "validation_context": trusted,
    }


def _c2_inputs():
    # Invalid C1 inputs intentionally exercise the real C2 rejection path.
    return {
        "inputs": [],
        "scope_declaration": {},
        "authorization_evidence": None,
        "prospective_commitment": {},
        "authorization_registration": None,
        "protected_holdout_events": None,
    }


def _decision(status="AUTHORIZATION_REQUIRED"):
    result = {
        "assessment_version": C2_VERSION,
        "status": status,
        "reasons": [],
        "scope_digest": "sha256:" + "a" * 64,
        "commitment_digest": None,
        "stages": {},
        "scientific_alignment_verified": False,
        "scope_binding_verified": False,
        "signature_verified": False,
        "issuer_trusted": False,
        "commitment_binding_verified": False,
        "signed_events_verified": False,
        "trusted_checkpoint_verified": False,
        "independent_time_provenance_verified": False,
        "access_completeness_verified": False,
        "prospective_authorization_verified": False,
        "authorization": "not_established",
        "authentication": "not_established",
        "comparison_permitted": False,
        "ranking_permitted": False,
    }
    result["decision_digest"] = _digest(result)
    return result


def _run(**overrides):
    args = {
        **_internal(),
        "benchmark_authorization_inputs": _c2_inputs(),
    }
    args.update(overrides)
    return reconcile_generalization_authorization(**args)


def _assert_no_permissions(output):
    assert output["holdout_assessed"] is False
    assert output["generalization_gap"] is None
    for key in (
        "protected_holdout_assessment_permitted",
        "protected_holdout_comparison_permitted",
        "protected_holdout_ranking_permitted",
        "protected_holdout_disclosure_permitted",
    ):
        assert output[key] is False


def test_real_c2_invalid_inputs_remain_rejected():
    output = _run()
    assert output["status"] == "AUTHORIZATION_REJECTED"
    assert output["benchmark_authorization_status"] == (
        "AUTHORIZATION_REJECTED"
    )
    assert output["internal_evidence_verified"] is True
    _assert_no_permissions(output)


def test_valid_but_insufficient_c2_decision_stays_required(monkeypatch):
    decision = _decision()

    monkeypatch.setattr(
        "zenerestimation.comparison.generalization_authorization_integration."
        "assess_independent_benchmark_authorization",
        lambda **kwargs: deepcopy(decision),
    )

    output = _run(supplied_authorization_decision=deepcopy(decision))
    assert output["status"] == "AUTHORIZATION_REQUIRED"
    assert output["selected_model"] == "GRU"
    assert output["benchmark_authorization_decision_digest"] == (
        decision["decision_digest"]
    )
    _assert_no_permissions(output)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(comparison_permitted=True),
        lambda d: d.update(ranking_permitted=True),
        lambda d: d.update(authorization="established"),
        lambda d: d.update(status="AUTHORIZED"),
        lambda d: d.update(decision_digest="sha256:" + "0" * 64),
        lambda d: d.update(scope_digest="sha256:" + "f" * 64),
    ],
)
def test_forged_supplied_decision_is_rejected(monkeypatch, change):
    decision = _decision()
    forged = deepcopy(decision)
    change(forged)

    monkeypatch.setattr(
        "zenerestimation.comparison.generalization_authorization_integration."
        "assess_independent_benchmark_authorization",
        lambda **kwargs: deepcopy(decision),
    )

    output = _run(supplied_authorization_decision=forged)
    assert output["status"] == "AUTHORIZATION_REJECTED"
    assert "SUPPLIED_AUTHORIZATION_DECISION_MISMATCH" in output["reasons"]
    _assert_no_permissions(output)


def test_recomputed_c2_permission_escalation_fails(monkeypatch):
    forged = _decision()
    forged["comparison_permitted"] = True
    forged["decision_digest"] = _digest({
        k: v for k, v in forged.items()
        if k != "decision_digest"
    })

    monkeypatch.setattr(
        "zenerestimation.comparison.generalization_authorization_integration."
        "assess_independent_benchmark_authorization",
        lambda **kwargs: deepcopy(forged),
    )

    with pytest.raises(ValueError, match="fail-closed"):
        _run()


def test_invalid_internal_selection_cannot_be_rescued(monkeypatch):
    monkeypatch.setattr(
        "zenerestimation.comparison.generalization_authorization_integration."
        "assess_independent_benchmark_authorization",
        lambda **kwargs: _decision(),
    )

    internal = _internal()
    internal["selection_artifact"]["selection_decision_record"][
        "decision"
    ]["selected_model"] = "LSTM"

    with pytest.raises((TypeError, ValueError)):
        _run(**internal)


def test_reconciliation_is_deterministic_and_read_only():
    internal = _internal()
    c2_inputs = _c2_inputs()
    artifact_before = deepcopy(internal["selection_artifact"])
    context_before = deepcopy(internal["validation_context"])
    c2_before = deepcopy(c2_inputs)

    first = _run(**internal, benchmark_authorization_inputs=c2_inputs)
    second = _run(**internal, benchmark_authorization_inputs=c2_inputs)

    assert first == second
    assert internal["selection_artifact"] == artifact_before
    assert internal["validation_context"] == context_before
    assert c2_inputs == c2_before


def test_unexpected_authorization_arguments_rejected():
    inputs = _c2_inputs()
    inputs["authorization_granted"] = True

    with pytest.raises(ValueError, match="arguments"):
        _run(benchmark_authorization_inputs=inputs)
