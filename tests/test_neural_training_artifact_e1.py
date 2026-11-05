"""E1 selection-only training artifact tests."""
import json
import pytest

from zenerestimation.optimization.candidate import CandidateResult
from zenerestimation.optimization.result import OptimizationResult
from zenerestimation.optimization.neural_artifact import (
    SCHEMA_VERSION, build_neural_training_artifact, write_neural_training_artifact,
)


def make_result(fold_training=None):
    candidate = CandidateResult(
        params={"window": 4, "units": 8, "epochs": 2},
        score=0.25,
        metrics={"rmse": 0.25},
        metadata={"fold_training": fold_training, "validation_steps": 2},
    )
    return OptimizationResult(
        model="LSTM", metric="rmse", best_params=dict(candidate.params),
        best_score=0.25, candidates=(candidate,),
        metadata={"budget": {"planned_epoch_folds": 4}},
    )


def test_schema_and_selection_metric():
    data = build_neural_training_artifact(make_result())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["selection_metric"] == "rmse"
    assert data["best_score"] == 0.25


def test_missing_history_is_null():
    assert build_neural_training_artifact(make_result())["candidates"][0]["fold_training"] is None


def test_fold_evidence_preserved():
    folds = [{"fold_index": 1, "history": {"loss": [1.0, 0.5]},
              "events": None, "diagnostics": {"completed_epochs": 2}}]
    data = build_neural_training_artifact(make_result(folds))
    assert data["candidates"][0]["fold_training"] == folds


def test_snapshot_does_not_share_mutable_fold_data():
    folds = [{"fold_index": 1, "history": {"loss": [1.0]}}]
    data = build_neural_training_artifact(make_result(folds))
    data["candidates"][0]["fold_training"][0]["history"]["loss"].append(0.1)
    assert folds[0]["history"]["loss"] == [1.0]


def test_round_trip(tmp_path):
    path = write_neural_training_artifact(make_result(), tmp_path / "training.json")
    assert json.loads(path.read_text(encoding="utf-8")) == build_neural_training_artifact(make_result())


def test_no_benchmark_fields():
    data = build_neural_training_artifact(make_result())
    assert "benchmark" not in json.dumps(data).lower()
    assert "actual" not in data["candidates"][0]


def test_invalid_result():
    with pytest.raises(TypeError):
        build_neural_training_artifact(object())


def test_invalid_fold_shape():
    with pytest.raises(ValueError, match="fold_training"):
        build_neural_training_artifact(make_result({"fold_index": 1}))


def test_invalid_extension(tmp_path):
    with pytest.raises(ValueError, match=".json"):
        write_neural_training_artifact(make_result(), tmp_path / "training.txt")
