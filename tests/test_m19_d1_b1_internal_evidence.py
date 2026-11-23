"""M19-D1-B1 focused unit tests (replay is stubbed; not end-to-end M18 tests)."""
from __future__ import annotations

from copy import deepcopy
from math import sqrt
from types import SimpleNamespace

import pytest

from zenerestimation.comparison import generalization_internal_evidence as subject


def _inputs():
    # Unequal fold RMSEs demonstrate mean-of-fold != pooled RMSE.
    errors = [[1.0, -1.0], [2.0, -4.0]]
    rows, fold_metrics = [], []
    for fold, block in enumerate(errors):
        fold_metrics.append({
            "rmse": sqrt(sum(e * e for e in block) / len(block)),
            "mae": sum(abs(e) for e in block) / len(block),
        })
        for step, error in enumerate(block, 1):
            rows.append({
                "fold_index": fold, "horizon_step": step,
                "date": f"2025-{3 * step:02d}-01",
                "alignment_status": "verified",
                "actual_microVolt": 10.0,
                "predicted_microVolt": 10.0 + error,
                "error_microVolt": error,
                "absolute_error_microVolt": abs(error),
                "squared_error_microVolt2": error * error,
            })
    mean_rmse = sum(x["rmse"] for x in fold_metrics) / 2
    mean_mae = sum(x["mae"] for x in fold_metrics) / 2
    flat = [e for block in errors for e in block]
    profile = {
        "profile_version": "m17_b2_v1",
        "provenance": {"scope": "internal_expanding_window", "model": "LSTM"},
        "alignment_summary": {
            "fully_verified": True, "positional_only_rows": 0,
            "total_rows": 4, "verified_rows": 4,
        },
        "by_date": rows,
        "overall": {
            "n": 4,
            "rmse_microVolt": sqrt(sum(e * e for e in flat) / 4),
            "mae_microVolt": sum(abs(e) for e in flat) / 4,
        },
    }
    candidate = SimpleNamespace(
        status="evaluated", params={"window": 4}, score=mean_rmse,
        metrics={"rmse": mean_rmse, "mae": mean_mae},
        metadata={"selection_metric": "rmse", "accuracy_profile": profile,
                  "fold_metrics": fold_metrics},
    )
    result = SimpleNamespace(model="LSTM", metric="rmse", candidates=[candidate],
                             best_params={"window": 4}, best_score=mean_rmse)
    artifact = {"selection_decision_record": {
        "decision": {"status": "SELECTED", "authorized": True,
                     "selected_model": "LSTM", "selection_metric": "rmse",
                     "selection_score": mean_rmse},
        "decision_digest": "fixture-digest",
    }}
    return artifact, {"LSTM": result}, {"scope": "internal_expanding_window"}


@pytest.fixture
def evidence(monkeypatch):
    # Explicitly isolate the D1 numerical layer. Real M18 replay is tested
    # by the existing internal_selection/internal_ranking suite.
    monkeypatch.setattr(subject, "verify_internal_selection_artifact",
                        lambda *args, **kwargs: {"status": "verified"})
    return _inputs()


def _verify(evidence):
    artifact, results, context = evidence
    return subject.verify_internal_generalization_evidence(
        selection_artifact=artifact, results=results,
        validation_context=context,
    )


def test_valid_internal_evidence_and_no_holdout(evidence):
    report = _verify(evidence)
    assert report["status"] == "INTERNAL_EVIDENCE_VERIFIED"
    assert report["selected_model"] == "LSTM"
    assert report["ranking_rmse_microVolt"] != report["pooled_rmse_microVolt"]
    assert report["holdout_assessed"] is False
    assert report["generalization_gap"] is None
    assert report["protected_holdout_comparison_permitted"] is False
    assert report["protected_holdout_ranking_permitted"] is False


def test_replay_failure_is_rejected(evidence, monkeypatch):
    monkeypatch.setattr(subject, "verify_internal_selection_artifact",
                        lambda *args, **kwargs: {"status": "rejected"})
    with pytest.raises(ValueError, match="verification"):
        _verify(evidence)


@pytest.mark.parametrize("status,authorized", [
    ("PENDING_AUTHORIZATION", False),
    ("TIE_UNRESOLVED", True),
    ("SELECTED", False),
])
def test_nonselected_or_unauthorized_decision_is_rejected(evidence, status, authorized):
    evidence[0]["selection_decision_record"]["decision"].update(
        status=status, authorized=authorized,
    )
    with pytest.raises(ValueError, match="SELECTED"):
        _verify(evidence)


def test_tampered_fold_mae_is_rejected(evidence):
    evidence[1]["LSTM"].candidates[0].metadata["fold_metrics"][0]["mae"] += 0.5
    with pytest.raises(ValueError, match="fold\\[0\\].mae"):
        _verify(evidence)


def test_tampered_pooled_rmse_is_rejected(evidence):
    evidence[1]["LSTM"].candidates[0].metadata["accuracy_profile"]["overall"]["rmse_microVolt"] += 0.5
    with pytest.raises(ValueError, match="profile.overall.rmse"):
        _verify(evidence)


def test_tampered_observation_error_is_rejected(evidence):
    evidence[1]["LSTM"].candidates[0].metadata["accuracy_profile"]["by_date"][0]["error_microVolt"] += 0.5
    with pytest.raises(ValueError, match="error_microVolt"):
        _verify(evidence)


def test_positional_only_alignment_is_rejected(evidence):
    evidence[1]["LSTM"].candidates[0].metadata["accuracy_profile"]["by_date"][0]["alignment_status"] = "positional_only"
    with pytest.raises(ValueError, match="unverified accuracy observation"):
        _verify(evidence)


def test_duplicate_observation_is_rejected(evidence):
    rows = evidence[1]["LSTM"].candidates[0].metadata["accuracy_profile"]["by_date"]
    rows[1]["horizon_step"] = 1
    with pytest.raises(ValueError, match="duplicate accuracy observation"):
        _verify(evidence)


def test_inputs_are_not_mutated(evidence):
    before = deepcopy(evidence)
    _verify(evidence)
    assert evidence == before


def test_missing_selected_model_is_rejected(evidence):
    evidence[0]["selection_decision_record"]["decision"]["selected_model"] = "GRU"
    with pytest.raises(ValueError, match="missing from trusted results"):
        _verify(evidence)
