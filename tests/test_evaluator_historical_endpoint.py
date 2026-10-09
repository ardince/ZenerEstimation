"""Sprint 14.9G.2.3: historical holdout boundary regression tests."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.temporal import TemporalPreprocessor
from zenerestimation.evaluation.evaluator import ForecastEvaluator


@pytest.fixture
def dataset_110():
    dates = pd.date_range('1998-03-01', '2025-03-01', freq='QS-MAR')
    values = np.arange(len(dates), dtype=float) + 100.0
    missing = pd.to_datetime([
        '2002-09-01', '2010-03-01', '2010-06-01',
        '2024-06-01', '2024-09-01', '2024-12-01',
    ])
    values[dates.isin(missing)] = np.nan
    frame = pd.DataFrame({
        'ds': dates,
        'microVolt': values,
        'is_observed': ~dates.isin(missing),
    })
    return BatteryDataset(frame)


def test_110_historical_boundary_exact(dataset_110):
    evaluator = ForecastEvaluator(evaluation_steps=5, evaluation_end='2024-03-01')
    train, holdout = evaluator.split_dataset(dataset_110)
    assert train.data['ds'].iloc[-1] == pd.Timestamp('2022-12-01')
    assert holdout['ds'].tolist() == list(pd.date_range('2023-03-01', periods=5, freq='QS-MAR'))
    assert holdout['is_observed'].all()
    assert holdout['microVolt'].notna().all()
    assert train.data['ds'].max() < holdout['ds'].min()
    assert len(train.data) == 100
    assert len(holdout) == 5


def test_110_future_data_cannot_change_historical_split(dataset_110):
    evaluator = ForecastEvaluator(evaluation_steps=5, evaluation_end='2024-03-01')
    original_train, original_holdout = evaluator.split_dataset(dataset_110)
    changed = dataset_110.data.copy()
    changed.loc[changed['ds'] > '2024-03-01', 'microVolt'] = 999999.0
    train, holdout = evaluator.split_dataset(BatteryDataset(changed))
    pd.testing.assert_frame_equal(train.data.reset_index(drop=True), original_train.data.reset_index(drop=True))
    pd.testing.assert_frame_equal(holdout.reset_index(drop=True), original_holdout.reset_index(drop=True))


def test_110_interpolation_uses_training_only(dataset_110):
    evaluator = ForecastEvaluator(evaluation_steps=5, evaluation_end='2024-03-01')
    train, holdout = evaluator.split_dataset(dataset_110)
    prepared = TemporalPreprocessor().fit_transform(train)
    assert prepared.data['microVolt'].notna().all()
    assert prepared.data['ds'].max() == pd.Timestamp('2022-12-01')
    assert (prepared.data.loc[~prepared.data['is_observed'], 'microVolt'].notna()).all()
    assert holdout['microVolt'].notna().all()


def test_missing_holdout_rejected_before_model_fit(dataset_110):
    evaluator = ForecastEvaluator(
        evaluation_steps=5,
        evaluation_end='2024-03-01',
        preprocessor=TemporalPreprocessor(),
    )
    damaged = dataset_110.data.copy()
    damaged.loc[damaged['ds'] == '2023-09-01', 'microVolt'] = np.nan
    damaged.loc[damaged['ds'] == '2023-09-01', 'is_observed'] = False

    class NeverFit:
        def fit(self, dataset):
            pytest.fail('model must not fit with missing holdout targets')

        def predict(self, steps):
            pytest.fail('model must not predict with missing holdout targets')

    with pytest.raises(ValueError, match='missing target values'):
        evaluator.evaluate(BatteryDataset(damaged), NeverFit())


def test_default_no_endpoint_preserves_last_n_rows(dataset_110):
    evaluator = ForecastEvaluator(evaluation_steps=6)
    train, holdout = evaluator.split_dataset(dataset_110)
    pd.testing.assert_frame_equal(
        train.data.reset_index(drop=True),
        dataset_110.data.iloc[:-6].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        holdout.reset_index(drop=True),
        dataset_110.data.iloc[-6:].reset_index(drop=True),
    )


def test_endpoint_before_data_rejected(dataset_110):
    evaluator = ForecastEvaluator(evaluation_steps=5, evaluation_end='1997-12-01')
    with pytest.raises(ValueError, match='evaluation_end'):
        evaluator.split_dataset(dataset_110)


def test_endpoint_with_insufficient_history_rejected(dataset_110):
    evaluator = ForecastEvaluator(evaluation_steps=5, evaluation_end='1999-03-01')
    with pytest.raises(ValueError, match='more rows than'):
        evaluator.split_dataset(dataset_110)
