"""M20-C2-C: real Kalman through authorized bounded Random Search.

Additive integration tests; execute from the ZenerEstimation repository.
"""
from copy import deepcopy
import random

import numpy as np
import pandas as pd
import pytest

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
)
from zenerestimation.assessment.holdout_isolation import isolate_declared_holdout
from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.forecasting.kalman import KalmanForecaster
from zenerestimation.optimization.authorization import (
    LocalOptimizationCoordinator, OptimizationExecutionRequest,
)
from zenerestimation.optimization.authorized_execution_m20 import (
    ApprovedExecutionPlan, AuthorizedOptimizationRunner,
)
from zenerestimation.optimization.development_binding import build_validation_protocol_digest
from zenerestimation.optimization.internal_provenance import build_internal_validation_context
from zenerestimation.optimization.search_space_m20 import FiniteSearchSpace, candidate_digest
from zenerestimation.optimization.spaces import KalmanParameterSpace
from zenerestimation.optimization.validation import ExpandingWindowSplitter


def make_case(*, seed=7, max_trials=2, holdout_values=(100.0, 101.0), short_history=False):
    if short_history:
        development_values = [20.0 + 0.2 * i for i in range(9)]
        dates = pd.date_range('2020-03-01', periods=11, freq='QS-MAR')
        cutoff, forecast_dates, folds = '2022-03-01', ('2022-06-01', '2022-09-01'), 2
    else:
        t = np.arange(40, dtype=float)
        development_values = list(20.0 + 0.35 * t + 0.40 * np.sin(2.0 * np.pi * t / 4.0))
        dates = pd.date_range('2015-03-01', periods=42, freq='QS-MAR')
        cutoff, forecast_dates, folds = '2024-12-01', ('2025-03-01', '2025-06-01'), 3
    source = BatteryDataset(pd.DataFrame({
        'ds': dates, 'microVolt': development_values + list(holdout_values),
        'is_observed': [True] * len(dates),
    }))
    declaration = AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id='m20-c2-c-kalman-random-short' if short_history else 'm20-c2-c-kalman-random',
        assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id='410', development_cutoff=cutoff, forecast_dates=forecast_dates,
    )
    training = deepcopy(isolate_declared_holdout(source, declaration=declaration).development)
    splitter = ExpandingWindowSplitter(folds=folds, validation_steps=2)
    context = build_internal_validation_context(training, splitter, dataset_id='410')
    space = FiniteSearchSpace.from_legacy('Kalman', KalmanParameterSpace(
        process_noise=(1e-3, 2e-3, 3e-3), drift_noise=(1e-4,),
        regime_factor=(2.5,), regime_multiplier=(10.0,),
    ))
    assert space.candidate_count == 3
    request = OptimizationExecutionRequest(
        dataset_id='410', development_fingerprint=context['dataset_fingerprint'],
        development_cutoff=cutoff, model_family='Kalman', strategy='random',
        max_trials=max_trials, max_epoch_folds=None,
        validation_protocol_digest=build_validation_protocol_digest(context),
    )
    plan = ApprovedExecutionPlan(
        request_digest=request.digest, model_family='Kalman', strategy='random',
        selection_metric='rmse', seed=seed, search_space_digest=space.digest,
        validation_protocol_digest=request.validation_protocol_digest,
        model_factory_id='m20_c2_c.real_kalman.v1',
    )
    created_models = []

    def factory(**params):
        model = KalmanForecaster(dt=0.25, adaptive=True, **params)
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
        grant=grant, request=request, plan=plan, declaration=declaration,
        source_dataset=source, training_dataset=training, splitter=splitter,
        context=context, space=space, runner=runner, created_models=created_models,
        folds=folds, holdout_start=forecast_dates[0],
    )


def execute(case):
    return case['runner'].run(**{key: case[key] for key in (
        'grant', 'request', 'plan', 'declaration', 'source_dataset',
        'training_dataset', 'splitter',
    )})


def assert_verified_candidate(candidate, context, folds):
    assert candidate.status == 'evaluated'
    assert np.isfinite(candidate.score)
    assert candidate.metadata['internal_provenance'] == context
    assert candidate.metadata['selection_metric'] == 'rmse'
    fold_metrics = candidate.metadata['fold_metrics']
    assert len(fold_metrics) == folds
    assert candidate.score == pytest.approx(sum(f['rmse'] for f in fold_metrics) / folds)
    alignments = candidate.metadata['fold_alignment']
    assert len(alignments) == folds
    for alignment, expected_dates in zip(alignments, context['fold_validation_dates']):
        assert alignment['status'] == 'verified'
        assert alignment['timestamp_match'] is True
        assert pd.DatetimeIndex(alignment['forecast_dates']).equals(pd.DatetimeIndex(expected_dates))
        assert pd.DatetimeIndex(alignment['validation_dates']).equals(pd.DatetimeIndex(expected_dates))


def test_real_kalman_random_subset_alignment_ranking_fresh_models_and_single_use():
    case = make_case()
    outcome = execute(case)
    assert outcome.status == 'COMPLETED', outcome.failure
    assert outcome.failure is None
    assert outcome.optimization_result is not None
    snapshot = outcome.search_snapshot
    indices = random.Random(7).sample(range(3), 2)
    proposals = snapshot['proposals']
    assert [p['index'] for p in proposals] == indices
    assert len(set(indices)) == 2
    assert [p['sequence'] for p in proposals] == [1, 2]
    assert [p['params'] for p in proposals] == [case['space'].candidate_at(i) for i in indices]
    assert all(p['candidate_id'] == candidate_digest(p['params']) for p in proposals)
    assert len(outcome.candidate_results) == 2 < case['space'].candidate_count
    assert [c.params for c in outcome.candidate_results] == [p['params'] for p in proposals]
    assert snapshot['strategy'] == 'random'
    assert snapshot['seed'] == 7
    assert snapshot['max_trials'] == 2
    assert snapshot['proposed_count'] == snapshot['completed_count'] == 2
    assert snapshot['failed_count'] == 0
    assert snapshot['stop_reason'] == 'max_trials'
    assert snapshot['pending_sequence'] is None
    assert [f['status'] for f in snapshot['feedback']] == ['evaluated', 'evaluated']
    assert len(case['created_models']) == 2 * case['folds']
    assert len({id(model) for model in case['created_models']}) == 2 * case['folds']
    for model in case['created_models']:
        assert model.dt == 0.25
        assert model.adaptive is True
        assert len(model.dataset.target) >= 6
        assert pd.to_datetime(model.dataset.data['ds']).max() < pd.Timestamp(case['holdout_start'])
    for candidate in outcome.candidate_results:
        assert_verified_candidate(candidate, case['context'], case['folds'])
    best = min(outcome.candidate_results, key=lambda c: c.score)
    assert outcome.optimization_result.best_params == best.params
    assert outcome.optimization_result.best_score == pytest.approx(best.score)
    with pytest.raises(PermissionError):
        execute(case)


def test_real_kalman_random_same_seed_replays_proposals_and_scores():
    first = execute(make_case(seed=11))
    second = execute(make_case(seed=11))
    assert first.status == second.status == 'COMPLETED', (first.failure, second.failure)
    assert first.search_snapshot['proposals'] == second.search_snapshot['proposals']
    assert first.search_snapshot['feedback'] == second.search_snapshot['feedback']
    assert [c.params for c in first.candidate_results] == [c.params for c in second.candidate_results]
    assert [c.score for c in first.candidate_results] == pytest.approx(
        [c.score for c in second.candidate_results], rel=1e-8, abs=1e-8,
    )
    assert first.optimization_result.best_params == second.optimization_result.best_params


def test_real_kalman_random_protected_holdout_invariance():
    first = execute(make_case(seed=17, holdout_values=(100.0, 101.0)))
    second = execute(make_case(seed=17, holdout_values=(9000.0, -9000.0)))
    assert first.status == second.status == 'COMPLETED', (first.failure, second.failure)
    assert first.search_snapshot['proposals'] == second.search_snapshot['proposals']
    assert first.search_snapshot['feedback'] == second.search_snapshot['feedback']
    assert [c.score for c in first.candidate_results] == pytest.approx(
        [c.score for c in second.candidate_results], rel=1e-8, abs=1e-8,
    )
    assert first.optimization_result.best_params == second.optimization_result.best_params


def test_real_kalman_random_one_trial_budget_is_enforced():
    case = make_case(seed=23, max_trials=1)
    outcome = execute(case)
    assert outcome.status == 'COMPLETED', outcome.failure
    snapshot = outcome.search_snapshot
    assert len(outcome.candidate_results) == 1
    assert snapshot['proposed_count'] == snapshot['completed_count'] == 1
    assert snapshot['failed_count'] == 0
    assert snapshot['stop_reason'] == 'max_trials'
    assert snapshot['pending_sequence'] is None
    assert [p['index'] for p in snapshot['proposals']] == random.Random(23).sample(range(3), 1)
    assert len(case['created_models']) == case['folds']
    assert_verified_candidate(outcome.candidate_results[0], case['context'], case['folds'])
    assert outcome.optimization_result.best_params == outcome.candidate_results[0].params


def test_real_kalman_random_short_history_all_failed_bounded_and_single_use():
    case = make_case(short_history=True)
    outcome = execute(case)
    assert outcome.status == 'ALL_FAILED', outcome.failure
    assert outcome.failure is None
    assert outcome.optimization_result is None
    snapshot = outcome.search_snapshot
    indices = random.Random(7).sample(range(3), 2)
    assert [p['index'] for p in snapshot['proposals']] == indices
    assert len(outcome.candidate_results) == 2
    assert snapshot['proposed_count'] == snapshot['failed_count'] == 2
    assert snapshot['completed_count'] == 0
    assert snapshot['stop_reason'] == 'max_trials'
    assert snapshot['pending_sequence'] is None
    assert [f['status'] for f in snapshot['feedback']] == ['failed', 'failed']
    assert all(f['score'] is None for f in snapshot['feedback'])
    assert len(case['created_models']) == 2  # First fold fails for each candidate.
    for candidate in outcome.candidate_results:
        assert candidate.status == 'failed'
        assert candidate.score is None
        assert not candidate.metrics
    for model in case['created_models']:
        assert len(model.dataset.target) == 5
        assert pd.to_datetime(model.dataset.data['ds']).max() < pd.Timestamp(case['holdout_start'])
    with pytest.raises(PermissionError):
        execute(case)


def test_real_kalman_random_short_history_failure_ignores_holdout_values():
    first = execute(make_case(short_history=True, holdout_values=(100.0, 101.0)))
    second = execute(make_case(short_history=True, holdout_values=(9000.0, -9000.0)))
    assert first.status == second.status == 'ALL_FAILED'
    assert first.search_snapshot['proposals'] == second.search_snapshot['proposals']
    assert first.search_snapshot['feedback'] == second.search_snapshot['feedback']
    assert [(r.params, r.status, r.score) for r in first.candidate_results] == [
        (r.params, r.status, r.score) for r in second.candidate_results
    ]
