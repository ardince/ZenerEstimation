"""M20-C3-B: real ARIMA through authorized, bounded Bayesian Search.

Additive integration tests for the ZenerEstimation repository environment.
"""
from copy import deepcopy
import random
import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
)
from zenerestimation.assessment.holdout_isolation import isolate_declared_holdout
from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.forecasting.arima import ARIMAForecaster
from zenerestimation.optimization.authorization import (
    LocalOptimizationCoordinator, OptimizationExecutionRequest,
)
from zenerestimation.optimization.authorized_execution_m20 import (
    ApprovedExecutionPlan, AuthorizedOptimizationRunner,
)
from zenerestimation.optimization.development_binding import build_validation_protocol_digest
from zenerestimation.optimization.internal_provenance import build_internal_validation_context
from zenerestimation.optimization.search_space_m20 import FiniteSearchSpace, candidate_digest
from zenerestimation.optimization.spaces import ARIMAParameterSpace
from zenerestimation.optimization.validation import ExpandingWindowSplitter


def make_case(*, seed=7, max_trials=3, holdout_values=(100.0, 101.0)):
    t = np.arange(40, dtype=float)
    development_values = 20.0 + 0.35 * t + 0.40 * np.sin(2.0 * np.pi * t / 4.0)
    dates = pd.date_range('2015-03-01', periods=42, freq='QS-MAR')
    source = BatteryDataset(pd.DataFrame({
        'ds': dates,
        'microVolt': list(development_values) + list(holdout_values),
        'is_observed': [True] * 42,
    }))
    declaration = AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id='m20-c3-b-arima-bayesian',
        assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id='410',
        development_cutoff='2024-12-01',
        forecast_dates=('2025-03-01', '2025-06-01'),
    )
    training = deepcopy(isolate_declared_holdout(source, declaration=declaration).development)
    splitter = ExpandingWindowSplitter(folds=3, validation_steps=2)
    context = build_internal_validation_context(training, splitter, dataset_id='410')
    space = FiniteSearchSpace.from_legacy(
        'ARIMA', ARIMAParameterSpace(p=(0, 1, 2, 3), d=(1,), q=(0,))
    )
    assert space.candidate_count == 4
    request = OptimizationExecutionRequest(
        dataset_id='410',
        development_fingerprint=context['dataset_fingerprint'],
        development_cutoff='2024-12-01',
        model_family='ARIMA', strategy='bayesian',
        max_trials=max_trials, max_epoch_folds=None,
        validation_protocol_digest=build_validation_protocol_digest(context),
    )
    plan = ApprovedExecutionPlan(
        request_digest=request.digest, model_family='ARIMA', strategy='bayesian',
        selection_metric='rmse', seed=seed, search_space_digest=space.digest,
        validation_protocol_digest=request.validation_protocol_digest,
        model_factory_id='m20_c3_b.real_arima.v1',
    )
    coordinator = LocalOptimizationCoordinator()
    grant = coordinator.approve(request)
    runner = AuthorizedOptimizationRunner(coordinator)
    runner.approve_plan(
        grant=grant, request=request, plan=plan,
        parameter_space=space, model_factory=ARIMAForecaster,
    )
    return dict(
        grant=grant, request=request, plan=plan, declaration=declaration,
        source_dataset=source, training_dataset=training, splitter=splitter,
        context=context, space=space, runner=runner,
    )


def execute(case):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        return case['runner'].run(**{key: case[key] for key in (
            'grant', 'request', 'plan', 'declaration', 'source_dataset',
            'training_dataset', 'splitter',
        )})


def assert_verified_candidate(candidate, context):
    assert candidate.status == 'evaluated'
    assert np.isfinite(candidate.score)
    assert candidate.metadata['internal_provenance'] == context
    assert candidate.metadata['selection_metric'] == 'rmse'
    folds = candidate.metadata['fold_metrics']
    assert len(folds) == 3
    assert candidate.score == pytest.approx(sum(f['rmse'] for f in folds) / 3)
    alignments = candidate.metadata['fold_alignment']
    assert len(alignments) == 3
    for alignment, expected in zip(alignments, context['fold_validation_dates']):
        assert alignment['status'] == 'verified'
        assert alignment['timestamp_match'] is True
        assert pd.DatetimeIndex(alignment['forecast_dates']).equals(pd.DatetimeIndex(expected))
        assert pd.DatetimeIndex(alignment['validation_dates']).equals(pd.DatetimeIndex(expected))


def test_real_arima_bayesian_subset_alignment_ranking_budget_and_single_use():
    case = make_case()
    outcome = execute(case)
    assert outcome.status == 'COMPLETED', outcome.failure
    assert outcome.failure is None
    assert outcome.optimization_result is not None
    snapshot = outcome.search_snapshot
    proposals = snapshot['proposals']
    assert snapshot['strategy'] == 'bayesian'
    assert snapshot['seed'] == 7
    assert snapshot['max_trials'] == 3
    assert snapshot['proposed_count'] == snapshot['completed_count'] == 3
    assert snapshot['failed_count'] == 0
    assert snapshot['stop_reason'] == 'max_trials'
    assert snapshot['pending_sequence'] is None
    assert len(outcome.candidate_results) == 3 < case['space'].candidate_count
    indices = [p['index'] for p in proposals]
    assert len(set(indices)) == 3
    assert indices[0] == random.Random(7).randrange(case['space'].candidate_count)
    assert [p['sequence'] for p in proposals] == [1, 2, 3]
    assert [p['params'] for p in proposals] == [case['space'].candidate_at(i) for i in indices]
    assert all(p['candidate_id'] == candidate_digest(p['params']) for p in proposals)
    assert [c.params for c in outcome.candidate_results] == [p['params'] for p in proposals]
    assert [f['status'] for f in snapshot['feedback']] == ['evaluated'] * 3
    assert [f['candidate_id'] for f in snapshot['feedback']] == [p['candidate_id'] for p in proposals]
    assert [f['score'] for f in snapshot['feedback']] == pytest.approx(
        [c.score for c in outcome.candidate_results]
    )
    for candidate in outcome.candidate_results:
        assert_verified_candidate(candidate, case['context'])
    best = min(outcome.candidate_results, key=lambda c: c.score)
    assert outcome.optimization_result.best_params == best.params
    assert outcome.optimization_result.best_score == pytest.approx(best.score)
    with pytest.raises(PermissionError):
        execute(case)


def test_real_arima_bayesian_same_seed_replays_adaptive_sequence_and_scores():
    first = execute(make_case(seed=11))
    second = execute(make_case(seed=11))
    assert first.status == second.status == 'COMPLETED', (first.failure, second.failure)
    assert first.search_snapshot['proposals'] == second.search_snapshot['proposals']
    assert [f['status'] for f in first.search_snapshot['feedback']] == [
        f['status'] for f in second.search_snapshot['feedback']
    ]
    assert [c.score for c in first.candidate_results] == pytest.approx(
        [c.score for c in second.candidate_results], rel=1e-8, abs=1e-8
    )
    assert first.optimization_result.best_params == second.optimization_result.best_params


def test_real_arima_bayesian_protected_holdout_invariance():
    first = execute(make_case(seed=17, holdout_values=(100.0, 101.0)))
    second = execute(make_case(seed=17, holdout_values=(9000.0, -9000.0)))
    assert first.status == second.status == 'COMPLETED', (first.failure, second.failure)
    assert first.search_snapshot['proposals'] == second.search_snapshot['proposals']
    assert [c.score for c in first.candidate_results] == pytest.approx(
        [c.score for c in second.candidate_results], rel=1e-8, abs=1e-8
    )
    assert first.optimization_result.best_params == second.optimization_result.best_params


def test_real_arima_bayesian_one_trial_is_bounded_and_audited():
    case = make_case(seed=23, max_trials=1)
    outcome = execute(case)
    assert outcome.status == 'COMPLETED', outcome.failure
    snapshot = outcome.search_snapshot
    assert len(outcome.candidate_results) == 1
    assert snapshot['proposed_count'] == snapshot['completed_count'] == 1
    assert snapshot['failed_count'] == 0
    assert snapshot['stop_reason'] == 'max_trials'
    assert snapshot['pending_sequence'] is None
    assert [p['index'] for p in snapshot['proposals']] == [random.Random(23).randrange(4)]
    assert_verified_candidate(outcome.candidate_results[0], case['context'])
