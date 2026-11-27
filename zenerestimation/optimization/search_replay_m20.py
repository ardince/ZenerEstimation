"""M20-D-B1/B2: read-only, independent Grid and Random Search replay.

This checks consistency of supplied evidence, not its external authenticity,
model accuracy, or authorization validity. No model is trained or grant consumed.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any
import math
import random

from .authorization import OptimizationExecutionRequest
from .authorized_execution_m20 import ApprovedExecutionPlan, AuthorizedOptimizationOutcome
from .search_space_m20 import FiniteSearchSpace, NEURAL_FAMILIES, candidate_digest

SCHEMA_VERSION = 'm20-search-replay-verification-v1'
SUPPORTED_SNAPSHOT_VERSION = 'm20_b4_strategy_v1'


class SearchReplayVerificationError(ValueError):
    """Supplied search evidence violates the search replay contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SearchReplayVerificationError(message)


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _score(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _record(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == keys, f'invalid {label} structure')
    return value


def verify_grid_search_replay(
    *,
    request: OptimizationExecutionRequest,
    plan: ApprovedExecutionPlan,
    parameter_space: FiniteSearchSpace,
    outcome: AuthorizedOptimizationOutcome,
) -> dict[str, Any]:
    """Verify Grid Search history against the approved configuration.

    The supplied objects are never modified. For ABORTED outcomes, the last
    proposal may be pending, or the search may have stopped before the error.
    """
    return _verify_search_replay(
        request=request,
        plan=plan,
        parameter_space=parameter_space,
        outcome=outcome,
        strategy='grid',
    )


def verify_random_search_replay(
    *,
    request: OptimizationExecutionRequest,
    plan: ApprovedExecutionPlan,
    parameter_space: FiniteSearchSpace,
    outcome: AuthorizedOptimizationOutcome,
) -> dict[str, Any]:
    """Verify Random Search history against the approved seeded order."""
    return _verify_search_replay(
        request=request,
        plan=plan,
        parameter_space=parameter_space,
        outcome=outcome,
        strategy='random',
    )


def _verify_search_replay(
    *,
    request: OptimizationExecutionRequest,
    plan: ApprovedExecutionPlan,
    parameter_space: FiniteSearchSpace,
    outcome: AuthorizedOptimizationOutcome,
    strategy: str,
) -> dict[str, Any]:
    _require(type(request) is OptimizationExecutionRequest, 'invalid request type')
    _require(type(plan) is ApprovedExecutionPlan, 'invalid plan type')
    _require(type(parameter_space) is FiniteSearchSpace, 'invalid search space type')
    _require(type(outcome) is AuthorizedOptimizationOutcome, 'invalid outcome type')
    _require(request.strategy == plan.strategy == strategy,
             f'not a {strategy.title()} Search execution')
    _require(request.model_family == plan.model_family == parameter_space.model_family,
             'model family mismatch')
    _require(plan.request_digest == request.digest == outcome.request_digest,
             'request digest mismatch')
    _require(plan.digest == outcome.execution_plan_digest, 'execution plan digest mismatch')
    _require(plan.search_space_digest == parameter_space.digest, 'search space digest mismatch')
    _require(plan.validation_protocol_digest == request.validation_protocol_digest,
             'validation protocol mismatch')

    snapshot = deepcopy(outcome.search_snapshot)
    keys = {
        'version', 'request_digest', 'space_digest', 'strategy', 'seed', 'fold_count',
        'max_trials', 'max_epoch_folds', 'proposed_count', 'completed_count',
        'failed_count', 'reserved_epoch_folds', 'stop_reason', 'pending_sequence',
        'proposals', 'feedback',
    }
    snapshot = _record(snapshot, keys, 'search snapshot')
    _require(type(snapshot['version']) is str and bool(snapshot['version']),
             'invalid search snapshot version')
    _require(snapshot['version'] == SUPPORTED_SNAPSHOT_VERSION,
             'unsupported search snapshot version')
    _require(snapshot['request_digest'] == request.digest, 'snapshot request mismatch')
    _require(snapshot['space_digest'] == parameter_space.digest, 'snapshot space mismatch')
    _require(snapshot['strategy'] == strategy, 'snapshot strategy mismatch')
    _require(type(snapshot['seed']) is int and snapshot['seed'] == plan.seed,
             'snapshot seed mismatch')
    _require(_integer(snapshot['fold_count'], minimum=1), 'invalid fold count')
    _require(type(snapshot['max_trials']) is int and snapshot['max_trials'] == request.max_trials,
             'trial limit mismatch')
    _require(type(snapshot['max_epoch_folds']) is type(request.max_epoch_folds)
             and snapshot['max_epoch_folds'] == request.max_epoch_folds,
             'budget limit mismatch')

    proposals = snapshot['proposals']
    feedback = snapshot['feedback']
    candidates = outcome.candidate_results
    _require(type(proposals) is list and type(feedback) is list, 'invalid histories')
    trial_limit = min(request.max_trials, parameter_space.candidate_count)
    _require(len(proposals) <= trial_limit,
             'proposal limit exceeded')
    _require(len(feedback) == len(candidates), 'feedback / candidate count mismatch')
    _require(len(feedback) <= len(proposals), 'feedback exceeds proposals')
    _require(len(proposals) - len(feedback) in (0, 1), 'invalid outstanding proposal count')

    proposal_keys = {'sequence', 'index', 'candidate_id', 'params', 'planned_epoch_folds'}
    feedback_keys = {'sequence', 'candidate_id', 'status', 'score'}
    candidate_indices = (
        random.Random(plan.seed).sample(range(parameter_space.candidate_count), trial_limit)
        if strategy == 'random' else range(trial_limit)
    )
    reserved = 0
    for index, raw in enumerate(proposals):
        proposal = _record(raw, proposal_keys, f'proposal {index + 1}')
        candidate_index = candidate_indices[index]
        params = parameter_space.candidate_at(candidate_index)
        cost = (params['epochs'] * snapshot['fold_count']
                if request.model_family in NEURAL_FAMILIES else 0)
        _require(type(proposal['sequence']) is int and proposal['sequence'] == index + 1,
                 'proposal sequence mismatch')
        _require(type(proposal['index']) is int and proposal['index'] == candidate_index,
                 f'{strategy.title()} Search candidate index mismatch')
        _require(type(proposal['params']) is dict and proposal['params'] == params,
                 'candidate parameters mismatch')
        _require(type(proposal['candidate_id']) is str
                 and proposal['candidate_id'] == candidate_digest(params),
                 'candidate digest mismatch')
        _require(type(proposal['planned_epoch_folds']) is int
                 and proposal['planned_epoch_folds'] == cost,
                 'proposal cost mismatch')
        reserved += cost
        if request.max_epoch_folds is not None:
            _require(reserved <= request.max_epoch_folds, 'reserved budget exceeded')

    evaluated = failed = 0
    for index, raw in enumerate(feedback):
        item = _record(raw, feedback_keys, f'feedback {index + 1}')
        proposal = proposals[index]
        candidate = candidates[index]
        _require(type(item['sequence']) is int
                 and item['sequence'] == proposal['sequence'], 'feedback sequence mismatch')
        _require(type(item['candidate_id']) is str
                 and item['candidate_id'] == proposal['candidate_id'],
                 'feedback candidate mismatch')
        _require(type(candidate.params) is dict
                 and candidate_digest(candidate.params) == proposal['candidate_id']
                 and candidate.params == proposal['params'], 'result candidate mismatch')
        _require(type(item['status']) is str and item['status'] in ('evaluated', 'failed')
                 and candidate.status == item['status'], 'feedback status mismatch')
        if item['status'] == 'evaluated':
            _require(_score(item['score']) and _score(candidate.score)
                     and float(item['score']) == float(candidate.score),
                     'feedback score mismatch')
            _require(candidate.metrics.get(plan.selection_metric) == candidate.score,
                     'candidate selection metric mismatch')
            evaluated += 1
        else:
            _require(item['score'] is None and candidate.score is None
                     and not candidate.metrics, 'failed candidate has metrics or score')
            failed += 1

    _require(type(snapshot['proposed_count']) is int
             and snapshot['proposed_count'] == len(proposals), 'proposed count mismatch')
    _require(type(snapshot['completed_count']) is int
             and snapshot['completed_count'] == evaluated, 'completed count mismatch')
    _require(type(snapshot['failed_count']) is int
             and snapshot['failed_count'] == failed, 'failed count mismatch')
    _require(type(snapshot['reserved_epoch_folds']) is int
             and snapshot['reserved_epoch_folds'] == reserved, 'reserved cost mismatch')

    pending = len(proposals) > len(feedback)
    if pending:
        _require(type(snapshot['pending_sequence']) is int
                 and snapshot['pending_sequence'] == len(proposals),
                 'pending sequence mismatch')
    else:
        _require(snapshot['pending_sequence'] is None, 'pending sequence mismatch')
    _require(not pending or outcome.status == 'ABORTED',
             'non-aborted outcome has pending proposal')

    stop = snapshot['stop_reason']
    _require(stop is None or type(stop) is str, 'invalid stop reason')
    if stop is not None:
        _require(not pending, 'stopped search cannot have pending proposal')
        # The engine checks max_trials before space_exhausted.
        if len(proposals) >= request.max_trials:
            expected_stop = 'max_trials'
        elif len(proposals) == parameter_space.candidate_count:
            expected_stop = 'space_exhausted'
        else:
            next_params = parameter_space.candidate_at(candidate_indices[len(proposals)])
            next_cost = (next_params['epochs'] * snapshot['fold_count']
                         if request.model_family in NEURAL_FAMILIES else 0)
            expected_stop = ('max_epoch_folds'
                             if request.max_epoch_folds is not None
                             and reserved + next_cost > request.max_epoch_folds else None)
        _require(stop == expected_stop, 'termination reason mismatch')

    if outcome.status != 'ABORTED':
        _require(not pending and stop is not None, 'finished outcome lacks termination')
        _require(evaluated > 0 if outcome.status == 'COMPLETED' else evaluated == 0,
                 'outcome status / successful candidates mismatch')
    else:
        _require(outcome.failure is not None, 'aborted outcome lacks failure evidence')
        _require(pending or stop is None or stop in
                 ('max_trials', 'space_exhausted', 'max_epoch_folds'),
                 'invalid aborted state')

    if outcome.status == 'COMPLETED':
        result = outcome.optimization_result
        _require(result is not None, 'missing optimization result')
        successful = [candidate for candidate in candidates if candidate.status == 'evaluated']
        best = min(successful, key=lambda candidate: candidate.score)
        _require(result.best_params == best.params and result.best_score == best.score,
                 'best candidate mismatch')
        _require(result.model == request.model_family and result.metric == plan.selection_metric,
                 'optimization result identity mismatch')
        _require(len(result.candidates) == len(candidates)
                 and all(a.to_dict() == b.to_dict()
                         for a, b in zip(result.candidates, candidates)),
                 'optimization result candidates mismatch')
    else:
        _require(outcome.optimization_result is None, 'unexpected optimization result')

    return {
        'schema_version': SCHEMA_VERSION,
        'status': 'VERIFIED',
        'strategy': strategy,
        'request_digest': request.digest,
        'execution_plan_digest': plan.digest,
        'search_space_digest': parameter_space.digest,
        'verified_proposals': len(proposals),
        'verified_feedback': len(feedback),
        'verified_reserved_epoch_folds': reserved,
        'outcome_status': outcome.status,
    }
