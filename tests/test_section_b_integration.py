"""Milestone 16.1 Section B: evaluator-level timestamp integration tests."""
import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.evaluation.evaluator import ForecastEvaluator
from zenerestimation.optimization.evaluator import OptimizationEvaluator
from zenerestimation.optimization.validation import ExpandingWindowSplitter
from types import SimpleNamespace


class QuarterlyModel:
    def __init__(self, shift=0):
        self.shift = shift

    def fit(self, dataset):
        self.last_date = pd.Timestamp(dataset.data['ds'].iloc[-1])
        self.last_value = float(dataset.data['microVolt'].iloc[-1])
        return self

    def predict(self, steps):
        dates = pd.date_range(self.last_date + pd.DateOffset(months=3), periods=steps, freq='QS-MAR')
        dates = dates + pd.DateOffset(months=self.shift)
        return SimpleNamespace(
            model='QuarterlyModel',
            forecast=pd.Series(np.repeat(self.last_value, steps)),
            horizon=steps,
            dates=pd.DatetimeIndex(dates),
        )


@pytest.fixture
def dataset():
    dates = pd.date_range('2020-03-01', periods=20, freq='QS-MAR')
    return BatteryDataset(pd.DataFrame({'ds': dates, 'microVolt': np.linspace(20, 30, len(dates))}))


def test_holdout_accepts_matching_dates(dataset):
    result = ForecastEvaluator(evaluation_steps=4).evaluate(dataset, QuarterlyModel())
    assert np.isfinite(result.rmse)


@pytest.mark.parametrize('shift', [-3, 3])
def test_holdout_rejects_shifted_dates(dataset, shift):
    with pytest.raises(ValueError, match='forecast dates do not match validation dates'):
        ForecastEvaluator(evaluation_steps=4).evaluate(dataset, QuarterlyModel(shift))


def test_internal_validation_accepts_matching_dates(dataset):
    evaluator = OptimizationEvaluator(ExpandingWindowSplitter(folds=2, validation_steps=4))
    result = evaluator.evaluate(dataset, QuarterlyModel, {'shift': 0})
    assert np.isfinite(result.score)


@pytest.mark.parametrize('shift', [-3, 3])
def test_internal_validation_rejects_shifted_dates(dataset, shift):
    evaluator = OptimizationEvaluator(ExpandingWindowSplitter(folds=2, validation_steps=4))
    with pytest.raises(ValueError, match='forecast dates do not match validation dates'):
        evaluator.evaluate(dataset, QuarterlyModel, {'shift': shift})
