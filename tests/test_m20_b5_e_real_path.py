"""M20-B5-E: production evaluator, partition binding, and timestamp integration.

Run in the ZenerEstimation repository; no evaluator or validator monkeypatches.
"""
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
)
from zenerestimation.assessment.holdout_isolation import isolate_declared_holdout
from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.forecasting.result import ForecastResult
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


class DeterministicQuarterlyModel:
    """Predict last training value + p; records each real fit partition."""
    def __init__(self, *, order, recorder, shift_dates=False):
        if type(order) is not tuple or len(order) != 3:
            raise ValueError('ARIMA order must be a (p, d, q) tuple')
        self.p = order[0]
        self.recorder = recorder
        self.shift_dates = shift_dates

    def fit(self, dataset):
        frame = dataset.data.copy(deep=True)
        self.recorder.append(frame)
        self.last = pd.Timestamp(frame['ds'].iloc[-1])
        self.value = float(frame['microVolt'].iloc[-1]) + self.p
        return self

    def predict(self, steps):
        dates = pd.date_range(start=self.last, periods=steps + 1, freq='QS-MAR')[1:]
        if self.shift_dates:
            dates = dates + pd.DateOffset(months=3)
        return ForecastResult(model='DeterministicQuarterlyModel',
                              forecast=pd.Series([self.value] * steps),
                              horizon=steps, dates=pd.DatetimeIndex(dates))


def make_case(*, shift_dates=False, holdout_values=(32.0, 33.0)):
    dates = pd.date_range('2021-03-01', periods=14, freq='QS-MAR')
    values = [float(i + 20) for i in range(12)] + list(holdout_values)
    source = BatteryDataset(pd.DataFrame({
        'ds': dates, 'microVolt': values, 'is_observed': [True] * 14,
    }))
    declaration = AssessmentDeclaration(
        contract_version=CONTRACT_VERSION, assessment_id='m20-b5-e-410',
        assessment_track=AssessmentTrack.RETROSPECTIVE, dataset_id='410',
        development_cutoff='2023-12-01',
        forecast_dates=('2024-03-01', '2024-06-01'),
    )
    training = deepcopy(isolate_declared_holdout(source, declaration=declaration).development)
    splitter = ExpandingWindowSplitter(folds=2, validation_steps=2)
    context = build_internal_validation_context(training, splitter, dataset_id='410')
    request = OptimizationExecutionRequest(
        dataset_id='410', development_fingerprint=context['dataset_fingerprint'],
        development_cutoff='2023-12-01', model_family='ARIMA', strategy='grid',
        max_trials=2, max_epoch_folds=None,
        validation_protocol_digest=build_validation_protocol_digest(context),
    )
    space = FiniteSearchSpace('ARIMA', ('p', 'd', 'q'), ((0, 1), (0,), (0,)), 2)
    plan = ApprovedExecutionPlan(
        request_digest=request.digest, model_family='ARIMA', strategy='grid',
        selection_metric='rmse', seed=7, search_space_digest=space.digest,
        validation_protocol_digest=request.validation_protocol_digest,
        model_factory_id='m20_b5_e.deterministic_model.v1',
    )
    recorder = []
    def factory(**params):
        return DeterministicQuarterlyModel(**params, recorder=recorder,
                                           shift_dates=shift_dates)
    coordinator = LocalOptimizationCoordinator()
    grant = coordinator.approve(request)
    runner = AuthorizedOptimizationRunner(coordinator)
    runner.approve_plan(grant=grant, request=request, plan=plan,
                        parameter_space=space, model_factory=factory)
    return dict(request=request, declaration=declaration, source_dataset=source,
                training_dataset=training, splitter=splitter, context=context,
                plan=plan, runner=runner, grant=grant, coordinator=coordinator,
                recorder=recorder)


def execute(case):
    return case['runner'].run(**{key: case[key] for key in (
        'grant', 'request', 'plan', 'declaration', 'source_dataset',
        'training_dataset', 'splitter')})


def test_real_evaluator_success_provenance_and_no_holdout_fit():
    case = make_case()
    outcome = execute(case)
    assert outcome.status == 'COMPLETED', outcome.failure
    assert len(outcome.candidate_results) == 2
    assert len(case['recorder']) == 4  # two proposals x two folds
    assert outcome.search_snapshot['pending_sequence'] is None
    assert outcome.optimization_result is not None
    assert outcome.optimization_result.best_score == min(
        result.score for result in outcome.candidate_results)
    for result in outcome.candidate_results:
        assert result.status == 'evaluated'
        assert result.metadata['internal_provenance'] == case['context']
        assert result.metadata['selection_metric'] == 'rmse'
        assert result.score == pytest.approx(sum(
            fold['rmse'] for fold in result.metadata['fold_metrics']) / 2)
        assert [entry['fold_index'] for entry in result.metadata['fold_alignment']] == [0, 1]
        for entry, expected in zip(result.metadata['fold_alignment'],
                                   case['context']['fold_validation_dates']):
            assert entry['status'] == 'verified'

            assert pd.DatetimeIndex(entry['validation_dates']).equals(
                pd.DatetimeIndex(expected)
            )
            assert pd.DatetimeIndex(entry['forecast_dates']).equals(
                pd.DatetimeIndex(expected)
            )

            assert entry['timestamp_match'] is True
    for frame in case['recorder']:
        assert frame['ds'].max() < pd.Timestamp('2023-12-01')
        assert not (frame['ds'] >= pd.Timestamp('2024-03-01')).any()
        assert frame['microVolt'].max() <= 31.0
    with pytest.raises(PermissionError):
        execute(case)


def test_real_validator_rejects_shifted_forecast_and_keeps_pending():
    case = make_case(shift_dates=True)
    outcome = execute(case)
    assert outcome.status == 'ABORTED'
    assert outcome.optimization_result is None
    assert outcome.search_snapshot['pending_sequence'] == 1
    assert outcome.candidate_results == ()
    assert outcome.failure is not None
    assert len(case['recorder']) == 1


def test_holdout_values_do_not_change_internal_candidate_scores():
    first = execute(make_case(holdout_values=(32.0, 33.0)))
    second = execute(make_case(holdout_values=(9000.0, -9000.0)))
    assert first.status == second.status == 'COMPLETED'
    assert [r.score for r in first.candidate_results] == [
        r.score for r in second.candidate_results]
    assert first.optimization_result.best_params == second.optimization_result.best_params


def test_modified_development_rejected_before_grant_consumption():
    case = make_case()
    case['training_dataset'].data.loc[0, 'microVolt'] = 999.0
    with pytest.raises(ValueError):
        execute(case)
    assert case['coordinator'].verify(case['grant'], case['request'])
    assert case['recorder'] == []
