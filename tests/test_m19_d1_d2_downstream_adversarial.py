"""M19-D1-D2: adversarial separation of D1 assessment and B3/B4 scoring.

No production changes, model execution, or protected-partition access.
The B3/B4 chain fixture is controlled test evidence, not independent authorization.
"""
from copy import deepcopy

import pytest

from tests.test_m19_b3_b3_holdout_error_scoring import case  # noqa: F401
from tests.test_m19_b3_b4_evaluation_evidence import chain  # noqa: F401
from tests.test_m19_d1_b2_b2_authorization_integration import (
    _internal, _c2_inputs, _decision,
)
from zenerestimation.assessment.holdout_error_scoring import score_disclosed_holdout
from zenerestimation.assessment.holdout_evaluation_evidence import (
    build_holdout_evaluation_evidence, verify_holdout_evaluation_evidence,
)
from zenerestimation.comparison.generalization_gap_assessment import (
    assess_generalization_gap,
)

_DENIED = (
    "protected_holdout_assessment_permitted",
    "protected_holdout_comparison_permitted",
    "protected_holdout_ranking_permitted",
    "protected_holdout_disclosure_permitted",
)


def _run(**changes):
    args = {**_internal(), "benchmark_authorization_inputs": _c2_inputs()}
    args.update(changes)
    return assess_generalization_gap(**args)


def _assert_closed(output):
    assert output["status"] in {"AUTHORIZATION_REQUIRED", "AUTHORIZATION_REJECTED"}
    assert output["internal_evidence_verified"] is True
    assert output["comparability_verified"] is False
    assert output["holdout_assessed"] is False
    assert output["generalization_gap"] is None
    assert output["generalization_gap_metric"] is None
    assert all(output[key] is False for key in _DENIED)
    assert "observed_microVolt" not in output
    assert "by_date" not in output
    assert "metrics" not in output


def test_real_c2_rejection_cannot_promote_b4_evidence(chain):
    b4 = build_holdout_evaluation_evidence(**chain)
    assert verify_holdout_evaluation_evidence(b4, **chain)
    assert b4["scientific_independence"] == "not_established"
    assert b4["authentication"] == "not_established"
    result = _run(supplied_authorization_decision=b4)
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert result["reasons"] == ["SUPPLIED_AUTHORIZATION_DECISION_MISMATCH"]
    _assert_closed(result)


@pytest.mark.parametrize("artifact_key", ["scoring_snapshot", "disclosure_snapshot", "eligibility_snapshot"])
def test_b3_b4_artifacts_cannot_be_supplied_as_c2_decision(chain, artifact_key):
    result = _run(supplied_authorization_decision=deepcopy(chain[artifact_key]))
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert result["reasons"] == ["SUPPLIED_AUTHORIZATION_DECISION_MISMATCH"]
    _assert_closed(result)


@pytest.mark.parametrize("injected", [
    {"disclosure_snapshot": {"observed_microVolt": [12.3]}},
    {"scoring_snapshot": {"status": "HOLDOUT_SCORED"}},
    {"protected_holdout_observations": [12.3]},
    {"authorization_granted": True},
    {"generalization_gap": 0.0},
])
def test_d1_api_rejects_direct_holdout_and_permission_injection(injected):
    with pytest.raises(TypeError):
        _run(**injected)


def test_b4_verified_evidence_does_not_create_comparability(chain):
    b4 = build_holdout_evaluation_evidence(**chain)
    assert b4["status"] == "HOLDOUT_EVALUATION_EVIDENCE_VERIFIED"
    assert b4["scientific_independence"] == "not_established"
    assert b4["authentication"] == "not_established"
    result = _run()
    assert result["status"] == "AUTHORIZATION_REJECTED"
    _assert_closed(result)


def test_matching_selected_model_name_still_does_not_bind_benchmark(monkeypatch):
    decision = _decision("AUTHORIZATION_REQUIRED")
    monkeypatch.setattr(
        "zenerestimation.comparison.generalization_authorization_integration."
        "assess_independent_benchmark_authorization",
        lambda **kwargs: deepcopy(decision),
    )
    result = _run(supplied_authorization_decision=deepcopy(decision))
    assert result["selected_model"] == "GRU"
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result["benchmark_scope_digest"] == decision["scope_digest"]
    _assert_closed(result)


def test_d1_never_calls_single_model_scoring_or_b4_builder(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("D1 must not execute B3/B4 scoring or evidence reconstruction")
    monkeypatch.setattr(
        "zenerestimation.assessment.holdout_error_scoring.score_disclosed_holdout",
        forbidden,
    )
    monkeypatch.setattr(
        "zenerestimation.assessment.holdout_evaluation_evidence.build_holdout_evaluation_evidence",
        forbidden,
    )
    _assert_closed(_run())


def test_d1_does_not_mutate_b3_b4_chain(chain):
    original = {
        key: value if key == "declaration" else deepcopy(value)
        for key, value in chain.items()
    }

def test_b3_scoring_rejects_internal_selection_as_eligibility(chain):
    internal = _internal()
    with pytest.raises((TypeError, ValueError, KeyError)):
        score_disclosed_holdout(
            declaration=chain["declaration"],
            eligibility_snapshot=internal["selection_artifact"],
            disclosure_snapshot=chain["disclosure_snapshot"],
        )
