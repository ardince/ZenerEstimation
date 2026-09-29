"""Contract tests for repeated-seed stability evaluation."""

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data import BatteryDataset
from zenerestimation.forecasting.result import ForecastResult

from zenerestimation.optimization import (
    ExpandingWindowSplitter,
    OptimizationEvaluator,
    StabilityEvaluator,
    StabilityResult,
)


# ---------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------


def _synthetic_dataset(n: int = 24) -> BatteryDataset:
    """Create a small deterministic quarterly dataset."""

    dates = pd.date_range(
        start="2018-03-01",
        periods=n,
        freq="QS-MAR",
    )

    t = np.arange(n, dtype=float)

    values = (
        20.0
        + 0.20 * t
        + 0.03 * np.sin(t / 2.0)
    )

    dataframe = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": values,
        }
    )

    return BatteryDataset(
        dataframe=dataframe,
    )


# ---------------------------------------------------------------------
# Lightweight deterministic fake forecaster
# ---------------------------------------------------------------------


class _SeedAwareForecaster:
    """Minimal forecaster used to test stability orchestration."""

    def __init__(
        self,
        *,
        offset: float,
        seed: int,
    ) -> None:
        self.offset = float(offset)
        self.seed = seed

        self._last_value = None
        self._drift = None
        self._last_date = None

    def fit(self, dataset):
        """Fit a simple deterministic level-and-drift model."""

        values = dataset.data["microVolt"].to_numpy(
            dtype=float
        )

        dates = pd.DatetimeIndex(
            dataset.data["ds"]
        )

        self._last_value = float(
            values[-1]
        )

        self._last_date = pd.Timestamp(
            dates[-1]
        )

        if len(values) >= 2:
            self._drift = float(
                values[-1] - values[-2]
            )
        else:
            self._drift = 0.0

        return self

    def predict(self, steps):
        """Produce a deterministic seed-dependent forecast."""

        seed_effect = (
            0.01 * self.seed
        )

        dates = pd.date_range(
            start=self._last_date
            + pd.offsets.QuarterBegin(
                startingMonth=3
            ),
            periods=steps,
            freq="QS-MAR",
        )

        values = np.asarray(
            [
                self._last_value
                + self._drift * step
                + self.offset
                + seed_effect
                for step in range(
                    1,
                    steps + 1,
                )
            ],
            dtype=float,
        )

        forecast = pd.Series(
            values,
            index=dates,
            name="microVolt",
        )

        return ForecastResult(
            model="SeedAwareForecaster",
            forecast=forecast,
            horizon=steps,
            dates=dates,
            metadata={
                "seed": self.seed,
                "offset": self.offset,
            },
        )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _optimization_evaluator(
    metric: str = "rmse",
) -> OptimizationEvaluator:
    """Construct the internal temporal evaluator."""

    splitter = ExpandingWindowSplitter(
        folds=2,
        validation_steps=3,
    )

    return OptimizationEvaluator(
        splitter,
        metric=metric,
    )


def _stability_evaluator(
    metric: str = "rmse",
) -> StabilityEvaluator:
    """Construct the repeated-seed evaluator."""

    return StabilityEvaluator(
        _optimization_evaluator(
            metric=metric,
        ),
        model="TestNeuralModel",
    )


def _model_factory(**params):
    """Create a fresh deterministic seed-aware model."""

    return _SeedAwareForecaster(
        offset=params["offset"],
        seed=params["seed"],
    )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_stability_evaluator_accepts_valid_configuration():
    """A valid OptimizationEvaluator and model name should be accepted."""

    evaluator = _optimization_evaluator()

    stability = StabilityEvaluator(
        evaluator,
        model="TestNeuralModel",
    )

    assert stability.evaluator is evaluator
    assert stability.model == "TestNeuralModel"


def test_stability_evaluator_rejects_invalid_evaluator():
    """The wrapped evaluator must be an OptimizationEvaluator."""

    with pytest.raises(TypeError):
        StabilityEvaluator(
            object(),
            model="TestNeuralModel",
        )


def test_stability_evaluator_rejects_nonstring_model():
    """Model identity must be a string."""

    with pytest.raises(TypeError):
        StabilityEvaluator(
            _optimization_evaluator(),
            model=123,
        )


@pytest.mark.parametrize(
    "model",
    [
        "",
        " ",
        "   ",
    ],
)
def test_stability_evaluator_rejects_empty_model(model):
    """Model identity must contain non-whitespace characters."""

    with pytest.raises(ValueError):
        StabilityEvaluator(
            _optimization_evaluator(),
            model=model,
        )


# ---------------------------------------------------------------------
# End-to-end evaluation
# ---------------------------------------------------------------------


def test_stability_evaluator_runs_end_to_end():
    """A frozen configuration should be evaluated across all seeds."""

    dataset = _synthetic_dataset()

    result = _stability_evaluator().evaluate(
        dataset,
        _model_factory,
        params={
            "offset": 0.0,
        },
        seeds=(0, 1, 2),
    )

    assert isinstance(
        result,
        StabilityResult,
    )

    assert result.model == "TestNeuralModel"
    assert result.metric == "rmse"

    assert result.params == {
        "offset": 0.0,
    }

    assert result.run_count == 3

    assert tuple(
        seed_result.seed
        for seed_result in result.seeds
    ) == (
        0,
        1,
        2,
    )


def test_stability_evaluator_preserves_seed_order():
    """Seed execution order must match the requested order."""

    dataset = _synthetic_dataset()

    result = _stability_evaluator().evaluate(
        dataset,
        _model_factory,
        params={
            "offset": 0.0,
        },
        seeds=(7, 3, 11),
    )

    assert tuple(
        seed_result.seed
        for seed_result in result.seeds
    ) == (
        7,
        3,
        11,
    )


def test_stability_evaluator_preserves_frozen_params():
    """Seed must not leak into the frozen parameter configuration."""

    dataset = _synthetic_dataset()

    params = {
        "offset": 0.25,
    }

    result = _stability_evaluator().evaluate(
        dataset,
        _model_factory,
        params=params,
        seeds=(0, 1, 2),
    )

    assert result.params == {
        "offset": 0.25,
    }

    assert "seed" not in result.params


def test_stability_evaluator_injects_seed_into_factory():
    """Each requested seed should be supplied to the model factory."""

    dataset = _synthetic_dataset()

    observed_seeds = []

    def tracking_factory(**params):
        observed_seeds.append(
            params["seed"]
        )

        return _SeedAwareForecaster(
            offset=params["offset"],
            seed=params["seed"],
        )

    _stability_evaluator().evaluate(
        dataset,
        tracking_factory,
        params={
            "offset": 0.0,
        },
        seeds=(2, 5, 9),
    )

    # 3 seeds × 2 temporal folds.
    assert observed_seeds == [
        2,
        2,
        5,
        5,
        9,
        9,
    ]


# ---------------------------------------------------------------------
# Fresh-model invariant
# ---------------------------------------------------------------------


def test_stability_evaluator_creates_fresh_model_per_seed_and_fold():
    """Every seed/fold combination must receive a fresh model."""

    dataset = _synthetic_dataset()

    created_models = []

    def tracking_factory(**params):
        model = _SeedAwareForecaster(
            offset=params["offset"],
            seed=params["seed"],
        )

        created_models.append(model)

        return model

    result = _stability_evaluator().evaluate(
        dataset,
        tracking_factory,
        params={
            "offset": 0.0,
        },
        seeds=(0, 1, 2),
    )

    assert result.run_count == 3

    # 3 seeds × 2 temporal folds.
    assert len(created_models) == 6

    assert len(
        {
            id(model)
            for model in created_models
        }
    ) == 6


# ---------------------------------------------------------------------
# Metrics and temporal-validation evidence
# ---------------------------------------------------------------------


def test_stability_evaluator_records_seed_metrics():
    """Every seed should contain the complete metric set."""

    dataset = _synthetic_dataset()

    result = _stability_evaluator().evaluate(
        dataset,
        _model_factory,
        params={
            "offset": 0.0,
        },
        seeds=(0, 1, 2),
    )

    for seed_result in result.seeds:
        assert set(
            seed_result.metrics
        ) == {
            "rmse",
            "mae",
            "mape",
        }

        assert np.isfinite(
            seed_result.metrics["rmse"]
        )

        assert np.isfinite(
            seed_result.metrics["mae"]
        )

        assert np.isfinite(
            seed_result.metrics["mape"]
        )

        assert np.isclose(
            seed_result.score,
            seed_result.metrics["rmse"],
        )


def test_stability_evaluator_preserves_temporal_validation_metadata():
    """Each seed should retain its temporal-validation evidence."""

    dataset = _synthetic_dataset()

    result = _stability_evaluator().evaluate(
        dataset,
        _model_factory,
        params={
            "offset": 0.0,
        },
        seeds=(0, 1),
    )

    for seed_result in result.seeds:
        validation = seed_result.metadata[
            "validation"
        ]

        assert validation[
            "validation"
        ] == "expanding_window"

        assert validation["folds"] == 2
        assert validation[
            "validation_steps"
        ] == 3

        assert validation[
            "selection_metric"
        ] == "rmse"

        assert len(
            validation["fold_metrics"]
        ) == 2


def test_stability_evaluator_summary_matches_seed_scores():
    """Stability summary statistics must use every seed score."""

    dataset = _synthetic_dataset()

    result = _stability_evaluator().evaluate(
        dataset,
        _model_factory,
        params={
            "offset": 0.0,
        },
        seeds=(0, 1, 2),
    )

    scores = np.asarray(
        [
            seed_result.score
            for seed_result in result.seeds
        ],
        dtype=float,
    )

    assert np.isclose(
        result.mean_score,
        np.mean(scores),
    )

    assert np.isclose(
        result.std_score,
        np.std(
            scores,
            ddof=0,
        ),
    )

    assert np.isclose(
        result.min_score,
        np.min(scores),
    )

    assert np.isclose(
        result.max_score,
        np.max(scores),
    )


def test_stability_evaluator_supports_mae():
    """The wrapped evaluator metric should define seed scores."""

    dataset = _synthetic_dataset()

    result = _stability_evaluator(
        metric="mae",
    ).evaluate(
        dataset,
        _model_factory,
        params={
            "offset": 0.0,
        },
        seeds=(0, 1),
    )

    assert result.metric == "mae"

    for seed_result in result.seeds:
        assert np.isclose(
            seed_result.score,
            seed_result.metrics["mae"],
        )


# ---------------------------------------------------------------------
# Stability metadata
# ---------------------------------------------------------------------


def test_stability_evaluator_records_analysis_metadata():
    """The result must explicitly describe repeated-seed analysis."""

    dataset = _synthetic_dataset()

    result = _stability_evaluator().evaluate(
        dataset,
        _model_factory,
        params={
            "offset": 0.0,
        },
        seeds=(0, 1, 2),
    )

    assert result.metadata[
        "analysis"
    ] == "repeated_seed"

    assert result.metadata[
        "seed_count"
    ] == 3

    assert result.metadata[
        "selection"
    ] == "none"


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_stability_evaluator_rejects_invalid_dataset():
    """Evaluation requires a BatteryDataset."""

    with pytest.raises(TypeError):
        _stability_evaluator().evaluate(
            object(),
            _model_factory,
            params={
                "offset": 0.0,
            },
            seeds=(0, 1),
        )


def test_stability_evaluator_rejects_noncallable_factory():
    """The model factory must be callable."""

    with pytest.raises(TypeError):
        _stability_evaluator().evaluate(
            _synthetic_dataset(),
            object(),
            params={
                "offset": 0.0,
            },
            seeds=(0, 1),
        )


def test_stability_evaluator_rejects_non_dictionary_params():
    """Frozen model parameters must be a dictionary."""

    with pytest.raises(TypeError):
        _stability_evaluator().evaluate(
            _synthetic_dataset(),
            _model_factory,
            params=[
                ("offset", 0.0),
            ],
            seeds=(0, 1),
        )


def test_stability_evaluator_rejects_empty_params():
    """The frozen model configuration must not be empty."""

    with pytest.raises(ValueError):
        _stability_evaluator().evaluate(
            _synthetic_dataset(),
            _model_factory,
            params={},
            seeds=(0, 1),
        )


def test_stability_evaluator_rejects_seed_inside_params():
    """Seed belongs to replication, not frozen model parameters."""

    with pytest.raises(ValueError):
        _stability_evaluator().evaluate(
            _synthetic_dataset(),
            _model_factory,
            params={
                "offset": 0.0,
                "seed": 42,
            },
            seeds=(0, 1),
        )


def test_stability_evaluator_rejects_empty_seeds():
    """At least one stability seed is required."""

    with pytest.raises(ValueError):
        _stability_evaluator().evaluate(
            _synthetic_dataset(),
            _model_factory,
            params={
                "offset": 0.0,
            },
            seeds=(),
        )


def test_stability_evaluator_rejects_duplicate_seeds():
    """Duplicate seeds must not count as stability replications."""

    with pytest.raises(ValueError):
        _stability_evaluator().evaluate(
            _synthetic_dataset(),
            _model_factory,
            params={
                "offset": 0.0,
            },
            seeds=(
                42,
                42,
            ),
        )


def test_stability_evaluator_rejects_boolean_seed():
    """Boolean values must not be accepted as seeds."""

    with pytest.raises(TypeError):
        _stability_evaluator().evaluate(
            _synthetic_dataset(),
            _model_factory,
            params={
                "offset": 0.0,
            },
            seeds=(
                True,
                1,
            ),
        )


def test_stability_evaluator_rejects_noninteger_seed():
    """Every stability seed must be an integer."""

    with pytest.raises(TypeError):
        _stability_evaluator().evaluate(
            _synthetic_dataset(),
            _model_factory,
            params={
                "offset": 0.0,
            },
            seeds=(
                0,
                1.5,
            ),
        )


def test_stability_evaluator_rejects_noniterable_seeds():
    """Seeds must be supplied as an iterable."""

    with pytest.raises(TypeError):
        _stability_evaluator().evaluate(
            _synthetic_dataset(),
            _model_factory,
            params={
                "offset": 0.0,
            },
            seeds=42,
        )