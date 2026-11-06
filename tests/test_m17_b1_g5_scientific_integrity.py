
"""M17-B1-G5: scientific provenance and numerical integrity.

Tests are written before implementing explicit dataset provenance
in OptimizationEvaluator.
"""

import math

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset

from zenerestimation.evaluation.accuracy_diagnostics import (
    build_accuracy_diagnostics,
)
from zenerestimation.optimization.evaluator import (
    OptimizationEvaluator,
)
from zenerestimation.optimization.validation import (
    ExpandingWindowSplitter,
)


from types import SimpleNamespace


def make_dataset():
    dates = pd.date_range(
        "2020-03-01",
        periods=20,
        freq="QS-MAR",
    )

    return BatteryDataset(
        pd.DataFrame({
            "ds": dates,
            "microVolt": np.arange(
                1, 21, dtype=float
            ),
        })
    )


class CountingForecaster:
    def __init__(self, offset=0.0, with_dates=True):
        self.offset = float(offset)
        self.with_dates = with_dates
        self.training = None

    def fit(self, dataset):
        self.training = dataset
        return self

    def predict(self, steps):
        value = (
            float(self.training.target.iloc[-1])
            + self.offset
        )

        result = SimpleNamespace(
            forecast=np.full(
                steps,
                value,
                dtype=float,
            )
        )

        if self.with_dates:
            last_date = pd.Timestamp(
                self.training.data["ds"].iloc[-1]
            )

            result.dates = pd.date_range(
                last_date + pd.DateOffset(months=3),
                periods=steps,
                freq="QS-MAR",
            )

        return result

def _fold(index, actual, predicted, status="verified"):
    return {
        "fold_index": index,
        "dates": pd.date_range(
            "2025-03-01", periods=len(actual), freq="QS-MAR"
        ),
        "actual": actual,
        "predicted": predicted,
        "alignment_status": status,
    }


def _report(folds):
    return build_accuracy_diagnostics(
        folds,
        dataset="732B-5610410",
        model="SyntheticModel",
        scope="internal_expanding_window",
    )


def test_g5_pooled_rmse_differs_from_mean_fold_rmse():
    report = _report([
        _fold(0, [10.0, 20.0], [11.0, 21.0]),
        _fold(1, [10.0, 20.0], [13.0, 23.0]),
    ])

    pooled = report["summary"]["rmse_microVolt"]
    mean_fold = report["summary"]["fold_stability"][
        "rmse_mean_microVolt"
    ]

    assert pooled == pytest.approx(math.sqrt(5.0))
    assert mean_fold == pytest.approx(2.0)
    assert pooled != pytest.approx(mean_fold)


def test_g5_mixed_alignment_evidence_is_preserved():
    report = _report([
        _fold(0, [10.0, 20.0], [11.0, 21.0], "verified"),
        _fold(
            1, [10.0, 20.0], [12.0, 22.0],
            "positional_only",
        ),
    ])

    assert report["provenance"]["alignment_statuses"] == [
        "verified",
        "positional_only",
    ]
    assert [row["alignment_status"] for row in report["rows"]] == [
        "verified",
        "verified",
        "positional_only",
        "positional_only",
    ]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_g5_invalid_measured_observation_is_rejected(invalid):
    with pytest.raises(ValueError):
        _report([
            _fold(0, [10.0, invalid], [11.0, 21.0]),
        ])


def test_g5_explicit_dataset_identifier_reaches_provenance():
    evaluator = OptimizationEvaluator(
        ExpandingWindowSplitter(folds=3, validation_steps=2),
        record_accuracy_diagnostics=True,
        accuracy_dataset_id="732B-5610410",
    )

    result = evaluator.evaluate(
        make_dataset(),
        CountingForecaster,
        {"offset": 0.0, "with_dates": True},
    )

    provenance = result.metadata[
        "accuracy_diagnostics"
    ]["provenance"]

    assert provenance["dataset"] == "732B-5610410"
    assert provenance["scope"] == "internal_expanding_window"


@pytest.mark.parametrize(
    "invalid_id",
    ["", "   ", 410, None],
)
def test_g5_invalid_explicit_dataset_identifier_is_rejected(
    invalid_id,
):
    with pytest.raises(
        (TypeError, ValueError),
        match="accuracy_dataset_id",
    ):
        OptimizationEvaluator(
            ExpandingWindowSplitter(
                folds=3,
                validation_steps=2,
            ),
            record_accuracy_diagnostics=True,
            accuracy_dataset_id=invalid_id,
        )


def test_g5_explicit_provenance_does_not_change_score():
    splitter = ExpandingWindowSplitter(
        folds=3,
        validation_steps=2,
    )

    baseline = OptimizationEvaluator(
        splitter,
        record_accuracy_diagnostics=True,
    ).evaluate(
        make_dataset(),
        CountingForecaster,
        {"offset": 0.0, "with_dates": True},
    )

    identified = OptimizationEvaluator(
        splitter,
        record_accuracy_diagnostics=True,
        accuracy_dataset_id="732B-5610410",
    ).evaluate(
        make_dataset(),
        CountingForecaster,
        {"offset": 0.0, "with_dates": True},
    )

    assert identified.score == pytest.approx(baseline.score)
    assert identified.metrics == pytest.approx(baseline.metrics)
    assert identified.metadata[
        "accuracy_diagnostics"
    ]["provenance"]["dataset"] == "732B-5610410"
