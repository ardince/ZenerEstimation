
"""M19-D1-B2-B1: real-artifact holdout-boundary acceptance tests."""

from copy import deepcopy

import pytest

from tests.test_m18_selection_audit_contract import (
    context,
    result,
    unique_results,
)
from zenerestimation.optimization.internal_selection_artifact import (
    build_internal_selection_artifact,
)
from zenerestimation.comparison.generalization_internal_evidence import (
    verify_internal_generalization_evidence,
)
from zenerestimation.comparison.generalization_holdout_boundary import (
    VERSION,
    assess_generalization_holdout_boundary,
)


def _inputs(*, authorized=True, results=None, trusted=None):
    if results is None:
        results = unique_results()
    if trusted is None:
        trusted = context()

    artifact = build_internal_selection_artifact(
        results,
        validation_context=trusted,
        authorized=authorized,
    )
    return artifact, results, trusted


def _assess(artifact, results, trusted):
    return assess_generalization_holdout_boundary(
        selection_artifact=artifact,
        results=results,
        validation_context=trusted,
    )


def test_b2_b1_real_artifact_requires_holdout_authorization():
    artifact, results, trusted = _inputs()
    output = _assess(artifact, results, trusted)

    assert output["assessment_version"] == VERSION
    assert output["status"] == "AUTHORIZATION_REQUIRED"
    assert output["reason"] == (
        "PROTECTED_HOLDOUT_AUTHORIZATION_NOT_ESTABLISHED"
    )
    assert output["internal_evidence_verified"] is True
    assert output["selected_model"] == "GRU"


def test_b2_b1_preserves_verified_selection_identity():
    artifact, results, trusted = _inputs()

    internal = verify_internal_generalization_evidence(
        selection_artifact=artifact,
        results=results,
        validation_context=trusted,
    )
    boundary = _assess(artifact, results, trusted)

    assert boundary["selected_model"] == internal["selected_model"]
    assert (
        boundary["selection_decision_digest"]
        == internal["selection_decision_digest"]
    )


def test_b2_b1_never_grants_protected_holdout_permissions():
    artifact, results, trusted = _inputs()
    output = _assess(artifact, results, trusted)

    assert output["holdout_assessed"] is False
    assert output["generalization_gap"] is None

    for permission in (
        "protected_holdout_assessment_permitted",
        "protected_holdout_comparison_permitted",
        "protected_holdout_ranking_permitted",
        "protected_holdout_disclosure_permitted",
    ):
        assert output[permission] is False


def test_b2_b1_is_deterministic_and_does_not_mutate_inputs():
    artifact, results, trusted = _inputs()

    original_artifact = deepcopy(artifact)
    original_context = deepcopy(trusted)
    original_results = {
        name: value.to_dict()
        for name, value in results.items()
    }

    first = _assess(artifact, results, trusted)
    second = _assess(artifact, results, trusted)

    assert first == second
    assert artifact == original_artifact
    assert trusted == original_context
    assert {
        name: value.to_dict()
        for name, value in results.items()
    } == original_results


def test_b2_b1_rejects_tampered_selection():
    artifact, results, trusted = _inputs()

    artifact["selection_decision_record"]["decision"][
        "selected_model"
    ] = "LSTM"

    with pytest.raises((TypeError, ValueError)):
        _assess(artifact, results, trusted)


def test_b2_b1_rejects_pending_selection():
    artifact, results, trusted = _inputs(authorized=False)

    with pytest.raises((TypeError, ValueError)):
        _assess(artifact, results, trusted)


def test_b2_b1_rejects_unresolved_tie():
    tied = {
        "LSTM": result("LSTM", rmse=(0.2, 0.4)),
        "GRU": result("GRU", rmse=(0.2, 0.4)),
    }
    artifact, results, trusted = _inputs(results=tied)

    with pytest.raises((TypeError, ValueError)):
        _assess(artifact, results, trusted)


def test_b2_b1_rejects_changed_trusted_context():
    artifact, results, trusted = _inputs()

    altered = context(
        dataset_id="different-development-dataset"
    )

    with pytest.raises((TypeError, ValueError)):
        _assess(artifact, results, altered)


def test_b2_b1_rejects_forged_authorization_claim():
    artifact, results, trusted = _inputs()

    artifact["protected_holdout_comparison_permitted"] = True
    artifact["protected_holdout_ranking_permitted"] = True

    with pytest.raises((TypeError, ValueError)):
        _assess(artifact, results, trusted)


def test_b2_b1_does_not_accept_holdout_or_authorization_inputs():
    artifact, results, trusted = _inputs()

    with pytest.raises(TypeError):
        assess_generalization_holdout_boundary(
            selection_artifact=artifact,
            results=results,
            validation_context=trusted,
            protected_holdout_observations=[1.0, 2.0],
        )

    with pytest.raises(TypeError):
        assess_generalization_holdout_boundary(
            selection_artifact=artifact,
            results=results,
            validation_context=trusted,
            authorization_granted=True,
        )
