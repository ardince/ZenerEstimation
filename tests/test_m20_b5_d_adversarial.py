"""M20-B5-D focused adversarial checks; uses the B5-C controlled fixture."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import pytest

from zenerestimation.optimization.candidate import CandidateResult
from zenerestimation.optimization.authorized_execution_m20 import (
    ApprovedExecutionPlan, AuthorizedOptimizationOutcome)
import zenerestimation.optimization.authorized_execution_m20 as b5
from tests.test_m20_b5_c_execution import setup, run  # noqa: F401


def _evaluator(s, *, alignments=None, score=1.):
    class Evaluator:
        def __init__(self, *args, **kwargs): pass
        def evaluate(self, *, dataset, model_factory, params):
            return CandidateResult(params=params, score=score, metrics={'rmse': score},
                metadata={'internal_provenance': deepcopy(s.context),
                          'fold_alignment': (alignments if alignments is not None else
                              [{'status': 'verified', 'fold_index': i} for i in range(2)]),
                          'selection_metric': 'rmse'})
    return Evaluator


def test_out_of_order_fold_evidence_aborts(setup, monkeypatch):
    s = setup
    monkeypatch.setattr(b5, 'OptimizationEvaluator', _evaluator(s, alignments=[
        {'status': 'verified', 'fold_index': 1}, {'status': 'verified', 'fold_index': 0}]))
    result = run(s)
    assert result.status == 'ABORTED'
    assert result.failure['category'] == 'CONTRACT_VIOLATION'
    assert result.search_snapshot['pending_sequence'] == 1


def test_missing_fold_index_aborts(setup, monkeypatch):
    s = setup
    monkeypatch.setattr(b5, 'OptimizationEvaluator', _evaluator(s, alignments=[
        {'status': 'verified'}, {'status': 'verified'}]))
    assert run(s).status == 'ABORTED'


def test_wrong_fold_count_rejected_before_consumption(setup, monkeypatch):
    s = setup
    monkeypatch.setattr(b5, 'build_internal_validation_context', lambda *a, **k: {
        **s.context, 'folds': 3})
    with pytest.raises(ValueError, match='fold count'):
        run(s)
    assert s.coordinator.verify(s.grant, s.request)


def test_model_factory_exception_becomes_failed_candidate(setup, monkeypatch):
    s = setup
    class Evaluator:
        def __init__(self, *args, **kwargs): pass
        def evaluate(self, *, dataset, model_factory, params):
            model_factory(**params)
    monkeypatch.setattr(b5, 'OptimizationEvaluator', Evaluator)
    # Replace the approved callable only in this controlled test to emulate
    # an approved factory whose constructor fails.
    old = s.runner._approved[id(s.grant)]
    def bad_factory(**params):
        raise RuntimeError('constructor failed')
    s.runner._approved[id(s.grant)] = (*old[:4], bad_factory, old[5])
    result = run(s)
    assert result.status == 'ALL_FAILED'
    assert result.search_snapshot['failed_count'] == 2


def test_unexpected_evaluator_error_aborts_not_candidate_failure(setup, monkeypatch):
    class Evaluator:
        def __init__(self, *args, **kwargs): pass
        def evaluate(self, *, dataset, model_factory, params):
            raise RuntimeError('evaluator invariant failed')
    monkeypatch.setattr(b5, 'OptimizationEvaluator', Evaluator)
    result = run(setup)
    assert result.status == 'ABORTED'
    assert result.search_snapshot['pending_sequence'] == 1
    assert result.candidate_results == ()


def test_result_score_metric_mismatch_aborts(setup, monkeypatch):
    s = setup
    class Evaluator:
        def __init__(self, *args, **kwargs): pass
        def evaluate(self, *, dataset, model_factory, params):
            return CandidateResult(params=params, score=1., metrics={'rmse': 2.},
                metadata={'internal_provenance': deepcopy(s.context),
                          'fold_alignment': [{'status': 'verified', 'fold_index': i} for i in range(2)],
                          'selection_metric': 'rmse'})
    monkeypatch.setattr(b5, 'OptimizationEvaluator', Evaluator)
    assert run(s).status == 'ABORTED'


def test_plan_seed_bool_rejected(setup):
    with pytest.raises(ValueError, match='seed'):
        replace(setup.plan, seed=True)


def test_outcome_status_invariants(setup):
    with pytest.raises(ValueError, match='completed'):
        AuthorizedOptimizationOutcome('COMPLETED', setup.request.digest,
            setup.plan.digest, {}, None, {}, (), None, None)
    with pytest.raises(ValueError, match='aborted'):
        AuthorizedOptimizationOutcome('ABORTED', setup.request.digest,
            setup.plan.digest, {}, None, {}, (), None, None)


def test_outcome_defensive_copy(setup, monkeypatch):
    s = setup
    monkeypatch.setattr(b5, 'OptimizationEvaluator', _evaluator(s))
    outcome = run(s)
    serialized = outcome.to_dict()
    serialized['candidate_results'][0]['params'].clear()
    serialized['grant_audit'].clear()
    assert outcome.to_dict()['candidate_results'][0]['params']
    assert outcome.to_dict()['grant_audit']


def test_plan_metric_substitution_rejected_before_consumption(setup):
    s = setup
    with pytest.raises(PermissionError):
        s.runner.run(grant=s.grant, request=s.request,
            plan=replace(s.plan, selection_metric='mae'), declaration=object(),
            source_dataset=object(), training_dataset=object(), splitter=s.splitter)
    assert s.coordinator.verify(s.grant, s.request)
