from types import SimpleNamespace

import pytest

from zenerestimation.optimization.neural_reporting import summarize_neural_candidates


def candidate(status="evaluated", metadata=None, score=0.4):
    return SimpleNamespace(status=status, params={"units": 8}, score=score,
                           metrics={"rmse": score}, metadata=metadata)


def result(*candidates, budget=None):
    return SimpleNamespace(model="LSTM", metric="rmse", best_params={"units": 8},
                           best_score=0.4, candidates=candidates,
                           metadata={"budget": budget} if budget else {})


def fold(index=1, onset=3, duration=1.5, plateau=False):
    return {"fold_index": index, "component": "LSTMForecaster", "status": "recorded",
            "history": {"epochs": []},
            "diagnostics": {"estimated_loss_improvement_onset_epoch": onset,
                            "best_training_loss_epoch": 4, "completed_epochs": 5,
                            "training_plateau_at_end": plateau},
            "events": {"status": "completed", "duration_seconds": duration}}


def test_two_fold_aggregation():
    report = summarize_neural_candidates(result(candidate(metadata={"fold_training": [fold(), fold(2, 5, 2.5, True)]})))
    row = report["candidates"][0]
    assert row["mean_estimated_onset_epoch"] == 4
    assert row["mean_recorded_duration_seconds"] == 2
    assert row["observed_completed_epoch_folds"] == 10
    assert row["plateau_folds"] == 1
    assert row["folds"][1]["fold_index"] == 2


def test_missing_diagnostics_are_not_invented():
    row = summarize_neural_candidates(result(candidate()))["candidates"][0]
    assert row["fold_count"] is None
    assert row["mean_estimated_onset_epoch"] is None
    assert row["observed_completed_epoch_folds"] is None


def test_missing_events_does_not_imply_zero_duration():
    item = fold()
    item["events"] = None
    row = summarize_neural_candidates(result(candidate(metadata={"fold_training": [item]})))["candidates"][0]
    assert row["recorded_duration_folds"] == 0
    assert row["mean_recorded_duration_seconds"] is None


def test_failed_candidate_is_reported():
    row = summarize_neural_candidates(result(candidate(status="failed", score=None)))["candidates"][0]
    assert row["status"] == "failed"
    assert row["score"] is None


def test_budget_is_retained_without_conflating_observed_work():
    report = summarize_neural_candidates(result(candidate(metadata={"fold_training": [fold()]}), budget={"planned_epoch_folds": 20}))
    assert report["budget"]["planned_epoch_folds"] == 20
    assert report["candidates"][0]["observed_completed_epoch_folds"] == 5


def test_report_does_not_mutate_source():
    c = candidate(metadata={"fold_training": [fold()]})
    r = result(c)
    report = summarize_neural_candidates(r)
    report["candidates"][0]["params"]["units"] = 999
    assert c.params["units"] == 8


def test_invalid_fold_training_rejected():
    with pytest.raises(ValueError):
        summarize_neural_candidates(result(candidate(metadata={"fold_training": "invalid"})))


def test_invalid_result_rejected():
    with pytest.raises(TypeError):
        summarize_neural_candidates(object())


def test_missing_onset_is_not_reported_as_epoch_zero():
    row = summarize_neural_candidates(result(candidate(metadata={"fold_training": [fold(onset=None)]})))["candidates"][0]
    assert row["mean_estimated_onset_epoch"] is None


def test_candidate_order_is_preserved():
    report = summarize_neural_candidates(result(candidate(score=0.9), candidate(score=0.4)))
    assert [c["score"] for c in report["candidates"]] == [0.9, 0.4]
