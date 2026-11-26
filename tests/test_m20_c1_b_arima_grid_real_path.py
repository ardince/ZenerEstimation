"""M20-C1-B: real ARIMA through the authorized, bounded grid runner.

Run from the ZenerEstimation repository with its development environment.
"""
from copy import deepcopy
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
from zenerestimation.optimization.search_space_m20 import FiniteSearchSpace
from zenerestimation.optimization.spaces import ARIMAParameterSpace
from zenerestimation.optimization.validation import ExpandingWindowSplitter


def make_case(*, holdout_values=(100.0, 101.0)):
    index = np.arange(40, dtype=float)
    development_values = 20.0 + 0.35 * index + 0.40 * np.sin(2.0 * np.pi * index / 4.0)
    dates = pd.date_range('2015-03-01', periods=42, freq='QS-MAR')
    source = BatteryDataset(pd.DataFrame({
        'ds': dates,
        'microVolt': list(development_values) + list(holdout_values),
        'is_observed': [True] * 42,
    }))
    declaration = AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id='m20-c1-b-arima-grid',
        assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id='410',
        development_cutoff='2024-12-01',
        forecast_dates=('2025-03-01', '2025-06-01'),
    )
    training = deepcopy(isolate_declared_holdout(source, declaration=declaration).development)
    splitter = ExpandingWindowSplitter(folds=3, validation_steps=2)
    context = build_internal_validation_context(training, splitter, dataset_id='410')
    legacy = ARIMAParameterSpace(p=(0, 1), d=(1,), q=(0,))
    space = FiniteSearchSpace.from_legacy('ARIMA', legacy)
    assert space.candidate_count == 2
    request = OptimizationExecutionRequest(
        dataset_id='410',
        development_fingerprint=context['dataset_fingerprint'],
        development_cutoff='2024-12-01',
        model_family='ARIMA', strategy='grid',
        max_trials=2, max_epoch_folds=None,
        validation_protocol_digest=build_validation_protocol_digest(context),
    )
    plan = ApprovedExecutionPlan(
        request_digest=request.digest, model_family='ARIMA', strategy='grid',
        selection_metric='rmse', seed=7, search_space_digest=space.digest,
        validation_protocol_digest=request.validation_protocol_digest,
        model_factory_id='m20_c1_b.real_arima.v1',
    )
    coordinator = LocalOptimizationCoordinator()
    grant = coordinator.approve(request)
    runner = AuthorizedOptimizationRunner(coordinator)
    runner.approve_plan(grant=grant, request=request, plan=plan,
                        parameter_space=space, model_factory=ARIMAForecaster)
    return dict(grant=grant, request=request, plan=plan,
                declaration=declaration, source_dataset=source,
                training_dataset=training, splitter=splitter,
                context=context, runner=runner, coordinator=coordinator)


def execute(case):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        return case['runner'].run(**{key: case[key] for key in (
            'grant', 'request', 'plan', 'declaration', 'source_dataset',
            'training_dataset', 'splitter',
        )})


def test_real_arima_authorized_grid_alignment_ranking_and_budget():
    case = make_case()
    outcome = execute(case)
    assert outcome.status == 'COMPLETED', outcome.failure
    assert outcome.failure is None
    assert outcome.optimization_result is not None
    assert len(outcome.candidate_results) == 2
    assert [c.params for c in outcome.candidate_results] == [
        {'order': (0, 1, 0)}, {'order': (1, 1, 0)},
    ]
    for candidate in outcome.candidate_results:
        assert candidate.status == 'evaluated'
        assert np.isfinite(candidate.score)
        assert candidate.metadata['internal_provenance'] == case['context']
        assert candidate.metadata['selection_metric'] == 'rmse'
        folds = candidate.metadata['fold_metrics']
        assert len(folds) == 3
        assert candidate.score == pytest.approx(sum(f['rmse'] for f in folds) / 3)
        for alignment, expected in zip(
            candidate.metadata['fold_alignment'],
            case['context']['fold_validation_dates'],
        ):
            assert alignment['status'] == 'verified'
            assert alignment['timestamp_match'] is True
            assert pd.DatetimeIndex(alignment['forecast_dates']).equals(pd.DatetimeIndex(expected))
            assert pd.DatetimeIndex(alignment['validation_dates']).equals(pd.DatetimeIndex(expected))
    best = min(outcome.candidate_results, key=lambda c: c.score)
    assert outcome.optimization_result.best_params == best.params
    assert outcome.optimization_result.best_score == pytest.approx(best.score)
    assert outcome.search_snapshot['pending_sequence'] is None
    with pytest.raises(PermissionError):
        execute(case)


def test_protected_holdout_values_do_not_change_real_arima_ranking():
    first = execute(make_case(holdout_values=(100.0, 101.0)))
    second = execute(make_case(holdout_values=(9000.0, -9000.0)))
    assert first.status == second.status == 'COMPLETED', (first.failure, second.failure)
    assert [c.params for c in first.candidate_results] == [c.params for c in second.candidate_results]
    assert [c.score for c in first.candidate_results] == pytest.approx(
        [c.score for c in second.candidate_results], abs=1e-8, rel=1e-8,
    )
    assert first.optimization_result.best_params == second.optimization_result.best_params
