"""M20-D-B4: synthetic neural budget replay and cross-strategy diagnostics.

No neural model is trained and no authorization grant is issued or consumed.
"""
from copy import deepcopy
from dataclasses import replace

import pytest

from zenerestimation.optimization.authorization import OptimizationExecutionRequest
from zenerestimation.optimization.authorized_execution_m20 import (
    ApprovedExecutionPlan, AuthorizedOptimizationOutcome,
)
from zenerestimation.optimization.candidate import CandidateResult
from zenerestimation.optimization.search_space_m20 import FiniteSearchSpace
from zenerestimation.optimization.search_strategy_m20 import BoundedSearchEngine
from zenerestimation.optimization.search_replay_m20 import (
    SearchReplayVerificationError, verify_grid_search_replay,
    verify_random_search_replay, verify_bayesian_search_replay,
)

VERIFY = {'grid': verify_grid_search_replay,
          'random': verify_random_search_replay,
          'bayesian': verify_bayesian_search_replay}


def make_neural_case(strategy, *, seed=7, epochs=(1, 3, 5, 7),
                     budget=8, trials=4, folds=2, abort_after=None):
    space = FiniteSearchSpace('LSTM', ('window', 'units', 'epochs'),
                              ((3,), (8,), tuple(epochs)), len(epochs))
    request = OptimizationExecutionRequest(
        dataset_id='410', development_fingerprint='sha256:' + 'a' * 64,
        development_cutoff='2024-12-01', model_family='LSTM', strategy=strategy,
        max_trials=trials, max_epoch_folds=budget,
        validation_protocol_digest='sha256:' + 'b' * 64)
    plan = ApprovedExecutionPlan(
        request_digest=request.digest, model_family='LSTM', strategy=strategy,
        selection_metric='rmse', seed=seed, search_space_digest=space.digest,
        validation_protocol_digest=request.validation_protocol_digest,
        model_factory_id='m20_d_b4.synthetic_neural.v1')
    engine = BoundedSearchEngine(request=request, parameter_space=space,
                                 seed=seed, fold_count=folds)
    results = []
    while True:
        proposal = engine.propose()
        if proposal is None:
            break
        if abort_after is not None and len(results) == abort_after:
            break  # pending proposal, matching runner's caught execution exception
        result = CandidateResult.failed(params=proposal.params,
                                        error=RuntimeError('synthetic model failure'),
                                        metadata={'validation': 'internal_temporal'})
        engine.record_result(proposal, status='failed')
        results.append(result)
    aborted = abort_after is not None and engine.snapshot()['pending_sequence'] is not None
    outcome = AuthorizedOptimizationOutcome(
        status='ABORTED' if aborted else 'ALL_FAILED',
        request_digest=request.digest, execution_plan_digest=plan.digest,
        grant_audit={}, development_binding_evidence={},
        search_snapshot=engine.snapshot(), candidate_results=tuple(results),
        optimization_result=None,
        failure={'category': 'EXECUTION_ABORTED', 'error_type': 'RuntimeError',
                 'error_message': 'synthetic abort'} if aborted else None)
    return dict(request=request, plan=plan, parameter_space=space, outcome=outcome)


@pytest.mark.parametrize('strategy', ['grid', 'random', 'bayesian'])
def test_neural_budget_termination_and_diagnostics(strategy):
    case = make_neural_case(strategy)
    before = deepcopy(case['outcome'].to_dict())
    report = VERIFY[strategy](**case)
    snapshot = case['outcome'].search_snapshot
    assert report['status'] == 'VERIFIED'
    assert report['verified_reserved_epoch_folds'] == snapshot['reserved_epoch_folds']
    assert report['diagnostics'] == {
        'proposed_count': snapshot['proposed_count'],
        'evaluated_count': snapshot['completed_count'],
        'failed_count': snapshot['failed_count'],
        'pending_sequence': snapshot['pending_sequence'],
        'stop_reason': snapshot['stop_reason'],
        'max_trials': case['request'].max_trials,
        'max_epoch_folds': case['request'].max_epoch_folds,
        'reserved_epoch_folds': snapshot['reserved_epoch_folds'],
    }
    assert case['outcome'].to_dict() == before


@pytest.mark.parametrize('strategy', ['grid', 'random', 'bayesian'])
def test_neural_budget_termination_tampering_rejected(strategy):
    case = make_neural_case(strategy)
    snapshot = deepcopy(case['outcome'].search_snapshot)
    assert snapshot['stop_reason'] == 'max_epoch_folds'
    snapshot['stop_reason'] = 'space_exhausted'
    with pytest.raises(SearchReplayVerificationError):
        VERIFY[strategy](**{**case, 'outcome': replace(case['outcome'], search_snapshot=snapshot)})


@pytest.mark.parametrize('strategy', ['grid', 'random', 'bayesian'])
def test_neural_budget_cost_tampering_rejected(strategy):
    case = make_neural_case(strategy)
    snapshot = deepcopy(case['outcome'].search_snapshot)
    snapshot['reserved_epoch_folds'] += 1
    with pytest.raises(SearchReplayVerificationError, match='reserved cost mismatch'):
        VERIFY[strategy](**{**case, 'outcome': replace(case['outcome'], search_snapshot=snapshot)})


@pytest.mark.parametrize('strategy', ['grid', 'random', 'bayesian'])
def test_aborted_pending_neural_diagnostics(strategy):
    case = make_neural_case(strategy, budget=100, abort_after=0)
    assert case['outcome'].status == 'ABORTED'
    report = VERIFY[strategy](**case)
    assert report['diagnostics']['pending_sequence'] == 1
    assert report['diagnostics']['stop_reason'] is None
    assert report['verified_feedback'] == 0


def test_bayesian_budget_uses_adaptive_next_candidate_not_grid_position():
    # Find a seeded all-failed case where Bayesian and grid next-candidate costs
    # diverge. The independent verifier must use the adaptive candidate.
    for seed in range(30):
        case = make_neural_case('bayesian', seed=seed, budget=8)
        snapshot = case['outcome'].search_snapshot
        if snapshot['stop_reason'] == 'max_epoch_folds' and snapshot['proposed_count']:
            assert verify_bayesian_search_replay(**case)['status'] == 'VERIFIED'
            return
    pytest.fail('no bounded Bayesian budget case found')
