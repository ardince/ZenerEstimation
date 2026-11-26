"""M20-C3-D: independent Bayesian numerical and adversarial contracts.

Additive tests for the bounded search engine. No real model fitting is needed:
C3-B/C3-C already cover real ARIMA and Kalman execution.
"""
import math
import random

import numpy as np
import pytest

from zenerestimation.optimization.authorization import OptimizationExecutionRequest
from zenerestimation.optimization.search_space_m20 import FiniteSearchSpace
from zenerestimation.optimization.search_strategy_m20 import (
    BAYES_POOL_LIMIT, BAYES_TRIAL_LIMIT, BoundedSearchEngine,
)
from zenerestimation.optimization.spaces import ARIMAParameterSpace


def make_engine(*, count=5, trials=4, seed=7):
    space = FiniteSearchSpace.from_legacy(
        'ARIMA', ARIMAParameterSpace(p=tuple(range(count)), d=(0,), q=(0,)),
    )
    request = OptimizationExecutionRequest(
        dataset_id='410', development_fingerprint='sha256:' + 'a' * 64,
        development_cutoff='2024-12-01', model_family='ARIMA',
        strategy='bayesian', max_trials=trials, max_epoch_folds=None,
        validation_protocol_digest='sha256:' + 'b' * 64,
    )
    return BoundedSearchEngine(request=request, parameter_space=space,
                               seed=seed, fold_count=3)


def independent_acquisition(space, observations, available):
    """Small independent scalar-loop GP reference; no production helpers."""
    def vector(idx):
        coords = []
        for dim in reversed(space.dimensions):
            idx, pos = divmod(idx, len(dim))
            coords.append(pos / max(len(dim) - 1, 1))
        return np.asarray(coords[::-1], dtype=float)

    def kernel(i, j):
        delta = vector(i) - vector(j)
        return math.exp(-0.5 * float(np.dot(delta, delta)))

    observed_indices = [i for i, _ in observations]
    scores = [float(s) for _, s in observations]
    scale = max(map(abs, scores))
    y = np.array([s / scale for s in scores] if scale else scores, dtype=float)
    gram = np.array([[kernel(i, j) for j in observed_indices]
                     for i in observed_indices], dtype=float)
    gram += 1e-6 * np.eye(len(observations))
    alpha = np.linalg.solve(gram, y)
    acquisition = []
    for i in available:
        cross = np.array([kernel(i, j) for j in observed_indices])
        mean = float(cross @ alpha)
        variance = max(0.0, 1.0 - float(cross @ np.linalg.solve(gram, cross)))
        acquisition.append(mean - math.sqrt(variance))
    return available[int(np.argmin(acquisition))]


def finish(engine, scores):
    proposals = []
    for score in scores:
        proposal = engine.propose()
        assert proposal is not None
        proposals.append(proposal)
        if score is None:
            engine.record_result(proposal, status='failed')
        else:
            engine.record_result(proposal, status='evaluated', score=score)
    return proposals


def test_d01_independent_gp_acquisition_after_each_successful_observation():
    engine = make_engine(count=7, trials=5)
    observations = []
    used = set()
    for score in (0.7, 0.2, 0.5, 0.1, 0.3):
        if observations:
            available = [i for i in range(7) if i not in used]
            expected = independent_acquisition(engine.space, observations, available)
        else:
            expected = random.Random(7).randrange(7)
        proposal = engine.propose()
        assert proposal.index == expected
        used.add(proposal.index)
        engine.record_result(proposal, status='evaluated', score=score)
        observations.append((proposal.index, score))
    assert engine.propose() is None
    assert engine.snapshot()['stop_reason'] == 'max_trials'


def test_d02_constant_zero_scores_and_exact_ties_are_deterministic():
    a, b = make_engine(count=5), make_engine(count=5)
    for _ in range(4):
        p, q = a.propose(), b.propose()
        assert p.index == q.index
        a.record_result(p, status='evaluated', score=0.0)
        b.record_result(q, status='evaluated', score=0.0)
    assert a.snapshot() == b.snapshot()
    assert len({p['index'] for p in a.snapshot()['proposals']}) == 4


def test_d03_extreme_finite_scores_remain_finite_and_auditable():
    engine = make_engine(count=6, trials=4)
    finish(engine, [1e308, -1e308, 1e-300, 0.0])
    assert engine.propose() is None
    snapshot = engine.snapshot()
    assert snapshot['completed_count'] == 4
    assert all(math.isfinite(f['score']) for f in snapshot['feedback'])


def test_d04_failed_feedback_consumes_budget_without_entering_surrogate():
    engine = make_engine(count=6, trials=4)
    first = engine.propose()
    assert first.index == random.Random(7).randrange(6)
    engine.record_result(first, status='failed')
    second = engine.propose()
    rng = random.Random(7)
    rng.randrange(6)
    remaining = [i for i in range(6) if i != first.index]
    assert second.index == remaining[rng.randrange(len(remaining))]
    engine.record_result(second, status='evaluated', score=0.4)
    available = [i for i in range(6) if i not in {first.index, second.index}]
    expected = independent_acquisition(engine.space, [(second.index, 0.4)], available)
    third = engine.propose()
    assert third.index == expected
    engine.record_result(third, status='failed')
    fourth = engine.propose()
    assert fourth.index not in {first.index, second.index, third.index}
    engine.record_result(fourth, status='failed')
    assert engine.propose() is None
    snap = engine.snapshot()
    assert (snap['proposed_count'], snap['completed_count'], snap['failed_count']) == (4, 1, 3)
    assert snap['stop_reason'] == 'max_trials'


def test_d05_surrogate_linear_solve_failure_does_not_reserve_trial(monkeypatch):
    engine = make_engine()
    finish(engine, [0.5])
    before = engine.snapshot()
    def broken_solve(*args, **kwargs):
        raise np.linalg.LinAlgError('injected')
    monkeypatch.setattr(np.linalg, 'solve', broken_solve)
    with pytest.raises(RuntimeError, match='Bayesian surrogate solve failed'):
        engine.propose()
    assert engine.snapshot() == before


def test_d06_nonfinite_acquisition_is_rejected_without_trial_reservation(monkeypatch):
    engine = make_engine()
    finish(engine, [0.5])
    before = engine.snapshot()
    original_solve = np.linalg.solve
    def nonfinite_solve(matrix, vector):
        result = original_solve(matrix, vector)
        return np.full_like(result, np.nan)
    monkeypatch.setattr(np.linalg, 'solve', nonfinite_solve)
    with pytest.raises(RuntimeError, match='nonfinite acquisition'):
        engine.propose()
    assert engine.snapshot() == before


def test_d07_pool_limit_and_trial_limit_enforced_at_construction():
    with pytest.raises(ValueError, match='candidate pool'):
        make_engine(count=BAYES_POOL_LIMIT + 1, trials=1)
    with pytest.raises(ValueError, match='trial limit'):
        make_engine(count=4, trials=BAYES_TRIAL_LIMIT + 1)
    engine = make_engine(count=4, trials=BAYES_TRIAL_LIMIT)
    finish(engine, [0.5] * 4)
    assert engine.propose() is None
    assert engine.snapshot()['stop_reason'] == 'space_exhausted'


def test_d08_pending_feedback_integrity_and_single_outstanding_proposal():
    engine = make_engine()
    proposal = engine.propose()
    with pytest.raises(RuntimeError, match='record feedback'):
        engine.propose()
    proposal.params['order'] = (999, 0, 0)
    with pytest.raises(ValueError, match='integrity mismatch'):
        engine.record_result(proposal, status='evaluated', score=1.0)
    assert engine.snapshot()['pending_sequence'] == 1
    assert engine.snapshot()['completed_count'] == 0


def test_d09_single_trial_budget_and_snapshot_accounting():
    engine = make_engine(count=6, trials=1)
    finish(engine, [None])
    assert engine.propose() is None
    snap = engine.snapshot()
    assert snap['proposed_count'] == 1
    assert snap['completed_count'] == 0
    assert snap['failed_count'] == 1
    assert snap['stop_reason'] == 'max_trials'
    assert snap['pending_sequence'] is None
