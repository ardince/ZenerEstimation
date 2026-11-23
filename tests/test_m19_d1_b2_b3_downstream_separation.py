
"""M19-D1-B2-B3: downstream holdout-boundary separation tests.

Verifies that existing single-model scoring remains intact and that
internal selection evidence cannot be promoted into holdout authorization.
"""

from copy import deepcopy

import pytest

# Reuse the accepted M19-B3 scoring and evaluation fixtures.
from tests.test_m19_b3_b3_holdout_error_scoring import case  # noqa: F401
from tests.test_m19_b3_b4_evaluation_evidence import chain  # noqa: F401

from tests.test_m18_selection_audit_contract import context, unique_results

from zenerestimation.optimization.internal_selection_artifact import (
    build_internal_selection_artifact,
)
from zenerestimation.assessment.holdout_error_scoring import (
    score_disclosed_holdout,
)
from zenerestimation.assessment.holdout_evaluation_evidence import (
    build_holdout_evaluation_evidence,
    verify_holdout_evaluation_evidence,
)
from zenerestimation.comparison.generalization_holdout_boundary import (
    assess_generalization_holdout_boundary,
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
    # Intentionally invalid C1 scope: real C2 rejection path.
    return {
        "inputs": [],
        "scope_declaration": {},
        "authorization_evidence": None,
        "prospective_commitment": {},
        "authorization_registration": None,
        "protected_holdout_events": None,
    }


def _assert_denied(result):
    assert result["holdout_assessed"] is False
    assert result["generalization_gap"] is None

    for key in (
        "protected_holdout_assessment_permitted",
        "protected_holdout_comparison_permitted",
        "protected_holdout_ranking_permitted",
        "protected_holdout_disclosure_permitted",
    ):
        assert result[key] is False


def test_existing_single_model_scoring_and_b4_remain_valid(chain):
    scoring = score_disclosed_holdout(
        declaration=chain["declaration"],
        eligibility_snapshot=chain["eligibility_snapshot"],
        disclosure_snapshot=chain["disclosure_snapshot"],
    )

    assert scoring == chain["scoring_snapshot"]
    assert scoring["status"] == "HOLDOUT_SCORED"
    assert scoring["scientific_independence"] == "not_established"

    evidence = build_holdout_evaluation_evidence(**chain)
    assert evidence["status"] == "HOLDOUT_EVALUATION_EVIDENCE_VERIFIED"
    assert evidence["scientific_independence"] == "not_established"
    assert evidence["authentication"] == "not_established"
    assert verify_holdout_evaluation_evidence(evidence, **chain)


def test_internal_selection_does_not_authorize_holdout():
    decision = assess_generalization_holdout_boundary(**_internal())

    assert decision["status"] == "AUTHORIZATION_REQUIRED"
    assert decision["internal_evidence_verified"] is True
    _assert_denied(decision)


def test_verified_b4_evidence_does_not_establish_independence(chain):
    evidence = build_holdout_evaluation_evidence(**chain)

    assert evidence["status"] == "HOLDOUT_EVALUATION_EVIDENCE_VERIFIED"
    assert evidence["scientific_independence"] == "not_established"
    assert evidence["authentication"] == "not_established"

    assert not any(
        key in evidence
        for key in (
            "protected_holdout_comparison_permitted",
            "protected_holdout_ranking_permitted",
            "prospective_authorization_verified",
        )
    )


@pytest.mark.parametrize(
    "unexpected",
    [
        {"protected_holdout_observations": [1.0, 2.0]},
        {"disclosure_snapshot": {"observed_microVolt": [1.0]}},
        {"scoring_snapshot": {"status": "HOLDOUT_SCORED"}},
        {"authorization_granted": True},
    ],
)
def test_reconciliation_rejects_holdout_inputs(unexpected):
    with pytest.raises(TypeError):
        reconcile_generalization_authorization(
            **_internal(),
            benchmark_authorization_inputs=_c2_inputs(),
            **unexpected,
        )


def test_reconciliation_cannot_promote_single_model_scoring(chain):
    c2_inputs = _c2_inputs()

    # A single-model scoring snapshot is not an independent C2
    # benchmark authorization decision.
    supplied = deepcopy(chain["scoring_snapshot"])

    output = reconcile_generalization_authorization(
        **_internal(),
        benchmark_authorization_inputs=c2_inputs,
        supplied_authorization_decision=supplied,
    )

    assert output["status"] == "AUTHORIZATION_REJECTED"
    assert "SUPPLIED_AUTHORIZATION_DECISION_MISMATCH" in output["reasons"]
    _assert_denied(output)


def test_internal_boundary_does_not_mutate_scoring_evidence(chain):
    internal = _internal()
    original = deepcopy(chain["scoring_snapshot"])

    first = assess_generalization_holdout_boundary(**internal)
    second = assess_generalization_holdout_boundary(**internal)

    assert first == second
    assert chain["scoring_snapshot"] == original
    _assert_denied(first)


def test_single_model_scoring_rejects_internal_selection_as_eligibility(chain):
    internal = _internal()

    with pytest.raises((TypeError, ValueError, KeyError)):
        score_disclosed_holdout(
            declaration=chain["declaration"],
            eligibility_snapshot=internal["selection_artifact"],
            disclosure_snapshot=chain["disclosure_snapshot"],
        )
