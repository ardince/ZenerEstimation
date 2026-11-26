"""M20-C4-B1: real LSTM through authorized bounded Grid Search.

Additive integration tests; run inside the ZenerEstimation repository.
These tests use real Keras training and the production evaluator/runner.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
)
from zenerestimation.assessment.holdout_isolation import isolate_declared_holdout
from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.forecasting.neural.lstm import LSTMForecaster
from zenerestimation.optimization.authorization import (
    LocalOptimizationCoordinator, OptimizationExecutionRequest,
)
from zenerestimation.optimization.authorized_execution_m20 import (
    ApprovedExecutionPlan, AuthorizedOptimizationRunner,
)
from zenerestimation.optimization.development_binding import build_validation_protocol_digest
from zenerestimation.optimization.internal_provenance import build_internal_validation_context
from zenerestimation.optimization.search_space_m20 import FiniteSearchSpace
from zenerestimation.optimization.spaces import LSTMParameterSpace
from zenerestimation.optimization.validation import ExpandingWindowSplitter


def make_case(*, holdout_values=(100.0, 101.0), epochs=(2, 3),
              max_trials=2, max_epoch_folds=10, development_length=40, window=3):
    t = np.arange(development_length, dtype=float)
    development_values = 20.0 + 0.35 * t + 0.40 * np.sin(2.0 * np.pi * t / 4.0)
    dates = pd.date_range('2015-03-01', periods=development_length + 2, freq='QS-MAR')
    cutoff = dates[development_length - 1].strftime('%Y-%m-%d')
    future_dates = tuple(d.strftime('%Y-%m-%d') for d in dates[-2:])
    source = BatteryDataset(pd.DataFrame({
        'ds': dates,
        'microVolt': list(development_values) + list(holdout_values),
        'is_observed': [True] * len(dates),
    }))
    declaration = AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id='m20-c4-b1-lstm-grid',
        assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id='410',
        development_cutoff=cutoff,
        forecast_dates=future_dates,
    )
    training = deepcopy(isolate_declared_holdout(source, declaration=declaration).development)
    splitter = ExpandingWindowSplitter(folds=2, validation_steps=2)
    context = build_internal_validation_context(training, splitter, dataset_id='410')
    legacy = LSTMParameterSpace(
        window=(window,), units=(4,), epochs=epochs, batch_size=(4,),
    )
    space = FiniteSearchSpace.from_legacy('LSTM', legacy)
    request = OptimizationExecutionRequest(
        dataset_id='410',
        development_fingerprint=context['dataset_fingerprint'],
        development_cutoff=cutoff,
        model_family='LSTM', strategy='grid',
        max_trials=max_trials, max_epoch_folds=max_epoch_folds,
        validation_protocol_digest=build_validation_protocol_digest(context),
    )
    plan = ApprovedExecutionPlan(
        request_digest=request.digest, model_family='LSTM', strategy='grid',
        selection_metric='rmse', seed=7, search_space_digest=space.digest,
        validation_protocol_digest=request.validation_protocol_digest,
        model_factory_id='m20_c4_b1.real_lstm.v1',
    )
    created_models = []

    def factory(**params):
        model = LSTMForecaster(**params, seed=42, record_training=True)
        created_models.append(model)
        return model

    coordinator = LocalOptimizationCoordinator()
    grant = coordinator.approve(request)
    runner = AuthorizedOptimizationRunner(coordinator)
    runner.approve_plan(
        grant=grant, request=request, plan=plan,
        parameter_space=space, model_factory=factory,
    )
    return dict(
        grant=grant, request=request, plan=plan,
        declaration=declaration, source_dataset=source,
        training_dataset=training, splitter=splitter,
        context=context, runner=runner, coordinator=coordinator,
        created_models=created_models,
    )


def execute(case):
    return case['runner'].run(**{key: case[key] for key in (
        'grant', 'request', 'plan', 'declaration', 'source_dataset',
        'training_dataset', 'splitter',
    )})


def test_real_lstm_authorized_grid_alignment_ranking_budget_and_fresh_models():
    case = make_case()
    outcome = execute(case)
    assert outcome.status == 'COMPLETED', outcome.failure
    assert outcome.failure is None
    assert outcome.optimization_result is not None
    assert len(outcome.candidate_results) == 2
    assert [c.params['epochs'] for c in outcome.candidate_results] == [2, 3]

    models = case['created_models']
    assert len(models) == 4  # 2 candidates x 2 folds
    assert len({id(m) for m in models}) == 4
    assert len({id(m.model) for m in models}) == 4
    assert len({id(m.scaler) for m in models}) == 4
    assert all(m.model is not None and m.history is not None for m in models)
    assert all(m.training_recorder is not None for m in models)
    for model in models:
        assert model.window == 3
        assert model.units == 4
        assert len(model.history.history['loss']) == model.epochs
        assert pd.to_datetime(model.dataset.data['ds']).max() < pd.Timestamp('2025-03-01')

    for candidate in outcome.candidate_results:
        assert candidate.status == 'evaluated'
        assert np.isfinite(candidate.score)
        assert candidate.metadata['internal_provenance'] == case['context']
        assert candidate.metadata['selection_metric'] == 'rmse'
        folds = candidate.metadata['fold_metrics']
        assert len(folds) == 2
        assert candidate.score == pytest.approx(sum(f['rmse'] for f in folds) / 2)
        alignments = candidate.metadata['fold_alignment']
        assert len(alignments) == 2
        for alignment, expected_dates in zip(alignments, case['context']['fold_validation_dates']):
            assert alignment['status'] == 'verified'
            assert alignment['timestamp_match'] is True
            assert pd.DatetimeIndex(alignment['forecast_dates']).equals(pd.DatetimeIndex(expected_dates))
            assert pd.DatetimeIndex(alignment['validation_dates']).equals(pd.DatetimeIndex(expected_dates))

    best = min(outcome.candidate_results, key=lambda c: c.score)
    assert outcome.optimization_result.best_params == best.params
    assert outcome.optimization_result.best_score == pytest.approx(best.score)
    snapshot = outcome.search_snapshot
    assert snapshot['proposed_count'] == 2
    assert snapshot['completed_count'] == 2
    assert snapshot['failed_count'] == 0
    assert snapshot['reserved_epoch_folds'] == 10
    assert [p['planned_epoch_folds'] for p in snapshot['proposals']] == [4, 6]
    assert snapshot['pending_sequence'] is None
    with pytest.raises(PermissionError):
        execute(case)


def test_lstm_epoch_fold_budget_stops_before_second_candidate():
    case = make_case(max_epoch_folds=4)
    outcome = execute(case)
    assert outcome.status == 'COMPLETED', outcome.failure
    assert [c.params['epochs'] for c in outcome.candidate_results] == [2]
    snapshot = outcome.search_snapshot
    assert snapshot['proposed_count'] == 1
    assert snapshot['reserved_epoch_folds'] == 4
    assert snapshot['stop_reason'] == 'max_epoch_folds'
    assert snapshot['pending_sequence'] is None
    assert len(case['created_models']) == 2


def test_protected_holdout_values_do_not_change_real_lstm_ranking():
    first = execute(make_case(holdout_values=(100.0, 101.0), epochs=(2,),
                              max_trials=1, max_epoch_folds=4))
    second = execute(make_case(holdout_values=(9000.0, -9000.0), epochs=(2,),
                               max_trials=1, max_epoch_folds=4))
    assert first.status == second.status == 'COMPLETED', (first.failure, second.failure)
    assert [c.params for c in first.candidate_results] == [c.params for c in second.candidate_results]
    assert [c.score for c in first.candidate_results] == pytest.approx(
        [c.score for c in second.candidate_results], rel=1e-5, abs=1e-5,
    )
    assert first.optimization_result.best_params == second.optimization_result.best_params


def test_insufficient_lstm_window_history_is_audited_all_failed():
    # Two folds of two validation points leave only six training points in
    # the first fold; window=8 cannot produce supervised training samples.
    case = make_case(development_length=10, window=8, epochs=(2,),
                     max_trials=1, max_epoch_folds=4)
    outcome = execute(case)
    assert outcome.status == 'ALL_FAILED', outcome.failure
    assert outcome.optimization_result is None
    assert outcome.failure is None
    assert len(outcome.candidate_results) == 1
    failed = outcome.candidate_results[0]
    assert failed.status == 'failed'
    assert failed.score is None
    assert not failed.metrics
    snapshot = outcome.search_snapshot
    assert snapshot['proposed_count'] == 1
    assert snapshot['completed_count'] == 0
    assert snapshot['failed_count'] == 1
    assert snapshot['reserved_epoch_folds'] == 4
    assert snapshot['pending_sequence'] is None
