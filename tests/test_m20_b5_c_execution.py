"""M20-B5-C integration contract tests. Run inside the ZenerEstimation repository."""
from copy import deepcopy
from types import SimpleNamespace
import pytest

from zenerestimation.optimization.authorization import (
    LocalOptimizationCoordinator, OptimizationExecutionRequest)
from zenerestimation.optimization.search_space_m20 import FiniteSearchSpace
from zenerestimation.optimization.candidate import CandidateResult
import zenerestimation.optimization.authorized_execution_m20 as b5


@pytest.fixture
def setup(monkeypatch):
    request = OptimizationExecutionRequest(
        dataset_id='test410', development_fingerprint='sha256:' + 'a'*64,
        development_cutoff='2023-12-01', model_family='ARIMA', strategy='grid',
        max_trials=2, max_epoch_folds=None,
        validation_protocol_digest='sha256:' + 'b'*64)
    space = FiniteSearchSpace('ARIMA', ('p', 'd', 'q'),
                              ((0, 1), (0,), (0,)), 2)
    plan = b5.ApprovedExecutionPlan(
        request_digest=request.digest, model_family='ARIMA', strategy='grid',
        selection_metric='rmse', seed=1, search_space_digest=space.digest,
        validation_protocol_digest=request.validation_protocol_digest,
        model_factory_id='trusted.factory.v1')
    coordinator = LocalOptimizationCoordinator()
    grant = coordinator.approve(request)
    runner = b5.AuthorizedOptimizationRunner(coordinator)
    factory = lambda **params: object()
    runner.approve_plan(grant=grant, request=request, plan=plan,
                        parameter_space=space, model_factory=factory)
    context = {'dataset_fingerprint': request.development_fingerprint,
               'folds': 2, 'fold_validation_dates': [['2022-01-01'], ['2023-01-01']]}
    monkeypatch.setattr(b5, 'verify_authorized_development_dataset',
                        lambda *args, **kwargs: SimpleNamespace(
                            development=SimpleNamespace(data='development-only'),
                            evidence={'status': 'DEVELOPMENT_BOUND'}))
    monkeypatch.setattr(b5, 'build_internal_validation_context',
                        lambda *args, **kwargs: deepcopy(context))
    monkeypatch.setattr(b5, 'build_validation_protocol_digest',
                        lambda *args, **kwargs: request.validation_protocol_digest)
    splitter = SimpleNamespace(split=lambda ds: (object(), object()))
    return SimpleNamespace(request=request, space=space, plan=plan,
                           coordinator=coordinator, grant=grant, runner=runner,
                           splitter=splitter, context=context)


def run(s):
    return s.runner.run(grant=s.grant, request=s.request, plan=s.plan,
                        declaration=object(), source_dataset=object(),
                        training_dataset=object(), splitter=s.splitter)


def test_success_and_single_use(setup, monkeypatch):
    s = setup
    class Evaluator:
        def __init__(self, *args, **kwargs):
            assert kwargs['require_verified_alignment'] is True
            assert kwargs['record_internal_provenance'] is True
            assert kwargs['accuracy_dataset_id'] == s.request.dataset_id
        def evaluate(self, *, dataset, model_factory, params):
            assert dataset.data == 'development-only'
            return CandidateResult(params=params, score=1., metrics={'rmse': 1.},
                metadata={'internal_provenance': deepcopy(s.context),
                          'fold_alignment': [{'status': 'verified', 'fold_index': i} for i in range(2)],
                          'selection_metric': 'rmse'})
    monkeypatch.setattr(b5, 'OptimizationEvaluator', Evaluator)
    outcome = run(s)
    assert outcome.status == 'COMPLETED'
    assert len(outcome.candidate_results) == 2
    assert outcome.optimization_result.best_score == 1.
    assert outcome.search_snapshot['pending_sequence'] is None
    with pytest.raises(PermissionError):
        run(s)
    audit = outcome.to_dict()
    audit['search_snapshot']['proposals'].clear()
    assert len(outcome.to_dict()['search_snapshot']['proposals']) == 2


def test_all_failed_is_auditable(setup, monkeypatch):
    class Evaluator:
        def __init__(self, *args, **kwargs): pass
        def evaluate(self, *, dataset, model_factory, params):
            raise b5.CandidateModelError(RuntimeError('training failed'))
    monkeypatch.setattr(b5, 'OptimizationEvaluator', Evaluator)
    outcome = run(setup)
    assert outcome.status == 'ALL_FAILED'
    assert outcome.optimization_result is None
    assert outcome.search_snapshot['failed_count'] == 2


def test_alignment_violation_aborts_with_pending_proposal(setup, monkeypatch):
    class Evaluator:
        def __init__(self, *args, **kwargs): pass
        def evaluate(self, *, dataset, model_factory, params):
            raise b5.ForecastAlignmentError('bad alignment', fold_index=0,
                                            observed_status='unverified')
    monkeypatch.setattr(b5, 'OptimizationEvaluator', Evaluator)
    outcome = run(setup)
    assert outcome.status == 'ABORTED'
    assert outcome.failure['category'] == 'CONTRACT_VIOLATION'
    assert outcome.search_snapshot['pending_sequence'] == 1
    assert outcome.candidate_results == ()


def test_provenance_mismatch_aborts(setup, monkeypatch):
    class Evaluator:
        def __init__(self, *args, **kwargs): pass
        def evaluate(self, *, dataset, model_factory, params):
            return CandidateResult(params=params, score=1., metrics={'rmse': 1.},
                metadata={'internal_provenance': {'wrong': True},
                          'fold_alignment': [{'status': 'verified', 'fold_index': i} for i in range(2)],
                          'selection_metric': 'rmse'})
    monkeypatch.setattr(b5, 'OptimizationEvaluator', Evaluator)
    outcome = run(setup)
    assert outcome.status == 'ABORTED'
    assert outcome.failure['category'] == 'CONTRACT_VIOLATION'


def test_wrong_plan_never_consumes_grant(setup):
    s = setup
    from dataclasses import replace
    wrong = replace(s.plan, selection_metric='mae')
    with pytest.raises(PermissionError):
        s.runner.run(grant=s.grant, request=s.request, plan=wrong,
                     declaration=object(), source_dataset=object(),
                     training_dataset=object(), splitter=s.splitter)
    assert s.coordinator.verify(s.grant, s.request)


def test_invalid_development_preflight_keeps_grant(setup, monkeypatch):
    s = setup
    def fail(*args, **kwargs):
        raise ValueError('wrong development fingerprint')
    monkeypatch.setattr(b5, 'verify_authorized_development_dataset', fail)
    with pytest.raises(ValueError, match='fingerprint'):
        run(s)
    assert s.coordinator.verify(s.grant, s.request)


def test_duplicate_plan_approval_rejected(setup):
    s = setup
    with pytest.raises(PermissionError):
        s.runner.approve_plan(grant=s.grant, request=s.request, plan=s.plan,
                              parameter_space=s.space, model_factory=lambda **kw: object())
