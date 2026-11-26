"""M20-C1-C: real Kalman forecasting through authorized bounded Grid Search.

Additive integration tests; run inside the ZenerEstimation repository.
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
from zenerestimation.forecasting.kalman import KalmanForecaster
from zenerestimation.optimization.authorization import (
    LocalOptimizationCoordinator, OptimizationExecutionRequest,
)
from zenerestimation.optimization.authorized_execution_m20 import (
    ApprovedExecutionPlan, AuthorizedOptimizationRunner,
)
from zenerestimation.optimization.development_binding import build_validation_protocol_digest
from zenerestimation.optimization.internal_provenance import build_internal_validation_context
from zenerestimation.optimization.search_space_m20 import FiniteSearchSpace
from zenerestimation.optimization.spaces import KalmanParameterSpace
from zenerestimation.optimization.validation import ExpandingWindowSplitter


def make_case(*, holdout_values=(100.0, 101.0)):
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
        assessment_id='m20-c1-c-kalman-grid',
        assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id='410',
        development_cutoff='2024-12-01',
        forecast_dates=('2025-03-01', '2025-06-01'),
    )
    training = deepcopy(isolate_declared_holdout(source, declaration=declaration).development)
    splitter = ExpandingWindowSplitter(folds=3, validation_steps=2)
    context = build_internal_validation_context(training, splitter, dataset_id='410')
    legacy = KalmanParameterSpace(
        process_noise=(1e-3, 2e-3),
        drift_noise=(1e-4,),
        regime_factor=(2.5,),
        regime_multiplier=(10.0,),
    )
    space = FiniteSearchSpace.from_legacy('Kalman', legacy)
    assert space.candidate_count == 2
    request = OptimizationExecutionRequest(
        dataset_id='410',
        development_fingerprint=context['dataset_fingerprint'],
        development_cutoff='2024-12-01',
        model_family='Kalman', strategy='grid',
        max_trials=2, max_epoch_folds=None,
        validation_protocol_digest=build_validation_protocol_digest(context),
    )
    plan = ApprovedExecutionPlan(
        request_digest=request.digest, model_family='Kalman', strategy='grid',
        selection_metric='rmse', seed=7, search_space_digest=space.digest,
        validation_protocol_digest=request.validation_protocol_digest,
        model_factory_id='m20_c1_c.real_kalman.v1',
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


def test_real_kalman_authorized_grid_alignment_ranking_budget_and_fresh_models():
    case = make_case()
    outcome = execute(case)
    assert outcome.status == 'COMPLETED', outcome.failure
    assert outcome.failure is None
    assert outcome.optimization_result is not None
    assert len(outcome.candidate_results) == 2
    expected = [
        {'process_noise': value, 'drift_noise': 1e-4,
         'regime_factor': 2.5, 'regime_multiplier': 10.0}
        for value in (1e-3, 2e-3)
    ]
    assert [candidate.params for candidate in outcome.candidate_results] == expected
    assert len(case['created_models']) == 6  # 2 candidates x 3 folds
    assert len({id(model) for model in case['created_models']}) == 6
    for model in case['created_models']:
        assert model.dt == 0.25
        assert model.adaptive is True
        assert model.dataset.target.index.size >= 6
        assert pd.to_datetime(model.dataset.data['ds']).max() < pd.Timestamp('2025-03-01')
    for candidate in outcome.candidate_results:
        assert candidate.status == 'evaluated'
        assert np.isfinite(candidate.score)
        assert candidate.metadata['internal_provenance'] == case['context']
        assert candidate.metadata['selection_metric'] == 'rmse'
        folds = candidate.metadata['fold_metrics']
        assert len(folds) == 3
        assert candidate.score == pytest.approx(sum(fold['rmse'] for fold in folds) / 3)
        alignments = candidate.metadata['fold_alignment']
        assert len(alignments) == 3
        for alignment, expected_dates in zip(alignments, case['context']['fold_validation_dates']):
            assert alignment['status'] == 'verified'
            assert alignment['timestamp_match'] is True
            assert pd.DatetimeIndex(alignment['forecast_dates']).equals(pd.DatetimeIndex(expected_dates))
            assert pd.DatetimeIndex(alignment['validation_dates']).equals(pd.DatetimeIndex(expected_dates))
    best = min(outcome.candidate_results, key=lambda candidate: candidate.score)
    assert outcome.optimization_result.best_params == best.params
    assert outcome.optimization_result.best_score == pytest.approx(best.score)
    assert outcome.search_snapshot['pending_sequence'] is None
    with pytest.raises(PermissionError):
        execute(case)


def test_protected_holdout_values_do_not_change_real_kalman_ranking():
    first = execute(make_case(holdout_values=(100.0, 101.0)))
    second = execute(make_case(holdout_values=(9000.0, -9000.0)))
    assert first.status == second.status == 'COMPLETED', (first.failure, second.failure)
    assert [candidate.params for candidate in first.candidate_results] == [
        candidate.params for candidate in second.candidate_results
    ]
    assert [candidate.score for candidate in first.candidate_results] == pytest.approx(
        [candidate.score for candidate in second.candidate_results], abs=1e-8, rel=1e-8,
    )
    assert first.optimization_result.best_params == second.optimization_result.best_params


def test_kalman_minimum_history_boundary_is_explicit():
    dates = pd.date_range('2020-03-01', periods=5, freq='QS-MAR')
    short_dataset = BatteryDataset(pd.DataFrame({
        'ds': dates, 'microVolt': [20.0, 20.2, 20.4, 20.6, 20.8],
    }))
    with pytest.raises(IndexError):
        KalmanForecaster().fit(short_dataset)


def test_kalman_search_space_rejects_invalid_noise_and_duplicate_values():
    with pytest.raises(ValueError):
        FiniteSearchSpace(
            'Kalman',
            ('process_noise', 'drift_noise', 'regime_factor', 'regime_multiplier'),
            ((0.0,), (1e-4,), (2.5,), (10.0,)), 1,
        )
    with pytest.raises(ValueError):
        FiniteSearchSpace(
            'Kalman',
            ('process_noise', 'drift_noise', 'regime_factor', 'regime_multiplier'),
            ((1e-3, 1e-3), (1e-4,), (2.5,), (10.0,)), 2,
        )
