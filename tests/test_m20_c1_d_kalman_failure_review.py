"""M20-C1 cross-contract: real Kalman minimum-history failures in authorized search."""
from copy import deepcopy

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
from zenerestimation.optimization.validation import ExpandingWindowSplitter


def make_short_history_case(*, holdout_values=(100.0, 101.0)):
    # Nine development quarters: with two folds of two quarters each,
    # the first expanding training partition has only five observations.
    dates = pd.date_range('2020-03-01', periods=11, freq='QS-MAR')
    source = BatteryDataset(pd.DataFrame({
        'ds': dates,
        'microVolt': [20.0 + 0.2 * i for i in range(9)] + list(holdout_values),
        'is_observed': [True] * 11,
    }))
    declaration = AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id='m20-c1-kalman-short-history',
        assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id='410',
        development_cutoff='2022-03-01',
        forecast_dates=('2022-06-01', '2022-09-01'),
    )
    training = deepcopy(isolate_declared_holdout(source, declaration=declaration).development)
    splitter = ExpandingWindowSplitter(folds=2, validation_steps=2)
    context = build_internal_validation_context(training, splitter, dataset_id='410')
    space = FiniteSearchSpace(
        'Kalman',
        ('process_noise', 'drift_noise', 'regime_factor', 'regime_multiplier'),
        ((1e-3, 2e-3), (1e-4,), (2.5,), (10.0,)), 2,
    )
    request = OptimizationExecutionRequest(
        dataset_id='410', development_fingerprint=context['dataset_fingerprint'],
        development_cutoff='2022-03-01', model_family='Kalman', strategy='grid',
        max_trials=2, max_epoch_folds=None,
        validation_protocol_digest=build_validation_protocol_digest(context),
    )
    plan = ApprovedExecutionPlan(
        request_digest=request.digest, model_family='Kalman', strategy='grid',
        selection_metric='rmse', seed=7, search_space_digest=space.digest,
        validation_protocol_digest=request.validation_protocol_digest,
        model_factory_id='m20_c1.review.real_kalman_short_history.v1',
    )
    models = []

    def factory(**params):
        model = KalmanForecaster(dt=0.25, adaptive=True, **params)
        models.append(model)
        return model

    coordinator = LocalOptimizationCoordinator()
    grant = coordinator.approve(request)
    runner = AuthorizedOptimizationRunner(coordinator)
    runner.approve_plan(
        grant=grant, request=request, plan=plan,
        parameter_space=space, model_factory=factory,
    )
    return locals()


def execute(case):
    return case['runner'].run(
        grant=case['grant'], request=case['request'], plan=case['plan'],
        declaration=case['declaration'], source_dataset=case['source'],
        training_dataset=case['training'], splitter=case['splitter'],
    )


def test_real_kalman_short_history_is_bounded_all_failed_not_aborted():
    case = make_short_history_case()
    outcome = execute(case)
    assert outcome.status == 'ALL_FAILED', outcome.failure
    assert outcome.failure is None
    assert outcome.optimization_result is None
    assert len(outcome.candidate_results) == 2  # Both authorized trials accounted for
    assert len(case['models']) == 2  # Each candidate fails on its first fold
    assert outcome.search_snapshot['pending_sequence'] is None
    for candidate in outcome.candidate_results:
        assert candidate.status == 'failed'
        assert candidate.score is None
        assert not candidate.metrics
    for model in case['models']:
        assert len(model.dataset.target) == 5
        assert pd.to_datetime(model.dataset.data['ds']).max() < pd.Timestamp('2022-06-01')
    with pytest.raises(PermissionError):
        execute(case)  # Grant consumed despite all model trials failing


def test_real_kalman_short_history_failure_ignores_protected_holdout_values():
    first = execute(make_short_history_case(holdout_values=(100.0, 101.0)))
    second = execute(make_short_history_case(holdout_values=(9000.0, -9000.0)))
    assert first.status == second.status == 'ALL_FAILED'
    assert [(r.params, r.status, r.score) for r in first.candidate_results] == [
        (r.params, r.status, r.score) for r in second.candidate_results
    ]
