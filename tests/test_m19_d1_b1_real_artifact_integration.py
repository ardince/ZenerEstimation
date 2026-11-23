"""M19-D1-B1-T13: real M18-C3 selection-artifact replay into D1-B1.

Uses canonical M18-C2 fixtures; intentionally does not monkeypatch the verifier.
No protected holdout inputs or model execution.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from tests.test_m18_selection_audit_contract import context, result, unique_results
from zenerestimation.optimization.internal_selection_artifact import (
    build_internal_selection_artifact,
    verify_internal_selection_artifact,
)
from zenerestimation.comparison.generalization_internal_evidence import (
    verify_internal_generalization_evidence,
)


def _assess(*, authorized=True, results=None, trusted=None, artifact=None):
    results = unique_results() if results is None else results
    trusted = context() if trusted is None else trusted
    artifact = (
        build_internal_selection_artifact(
            results, validation_context=trusted, authorized=authorized,
        )
        if artifact is None else artifact
    )
    return verify_internal_generalization_evidence(
        selection_artifact=artifact,
        results=results,
        validation_context=trusted,
    )


def test_t13_real_artifact_selected_gru_replays_and_reconstructs():
    results, trusted = unique_results(), context()
    artifact = build_internal_selection_artifact(
        results, validation_context=trusted, authorized=True,
    )
    assert verify_internal_selection_artifact(
        artifact, results=results, validation_context=trusted,
    )["status"] == "verified"
    output = _assess(results=results, trusted=trusted, artifact=artifact)
    assert output["status"] == "INTERNAL_EVIDENCE_VERIFIED"
    assert output["selected_model"] == "GRU"
    assert output["selection_replay_verified"] is True
    assert output["fold_count"] == len(trusted["fold_validation_dates"])
    assert output["observation_count"] == sum(map(len, trusted["fold_validation_dates"]))
    assert output["ranking_rmse_microVolt"] >= 0
    assert output["pooled_rmse_microVolt"] >= 0
    assert output["generalization_gap"] is None
    assert output["holdout_assessed"] is False
    assert output["protected_holdout_comparison_permitted"] is False
    assert output["protected_holdout_ranking_permitted"] is False


def test_t13_real_artifact_and_inputs_are_unchanged():
    results, trusted = unique_results(), context()
    artifact = build_internal_selection_artifact(
        results, validation_context=trusted, authorized=True,
    )
    before_artifact = deepcopy(artifact)
    before_trusted = deepcopy(trusted)
    before_results = {name: item.to_dict() for name, item in results.items()}
    first = _assess(results=results, trusted=trusted, artifact=artifact)
    second = _assess(results=results, trusted=trusted, artifact=artifact)
    assert first == second
    assert artifact == before_artifact
    assert trusted == before_trusted
    assert {name: item.to_dict() for name, item in results.items()} == before_results


def test_t13_real_artifact_rejects_tampering():
    results, trusted = unique_results(), context()
    artifact = build_internal_selection_artifact(
        results, validation_context=trusted, authorized=True,
    )
    artifact["selection_decision_record"]["decision"]["selected_model"] = "LSTM"
    with pytest.raises((TypeError, ValueError)):
        _assess(results=results, trusted=trusted, artifact=artifact)


def test_t13_real_artifact_rejects_pending_selection():
    with pytest.raises((TypeError, ValueError)):
        _assess(authorized=False)


def test_t13_real_artifact_rejects_unresolved_tie():
    results = {
        "LSTM": result("LSTM", rmse=(0.2, 0.4)),
        "GRU": result("GRU", rmse=(0.2, 0.4)),
    }
    with pytest.raises((TypeError, ValueError)):
        _assess(results=results, authorized=True)


def test_t13_real_artifact_rejects_changed_trusted_context():
    results, trusted = unique_results(), context()
    artifact = build_internal_selection_artifact(
        results, validation_context=trusted, authorized=True,
    )
    altered = context(dataset_id="different-development-dataset")
    with pytest.raises((TypeError, ValueError)):
        _assess(results=results, trusted=altered, artifact=artifact)
