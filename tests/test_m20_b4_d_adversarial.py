"""M20-B4-D adversarial acceptance: proposal-only; no training or holdout."""
from dataclasses import replace

import pytest

from zenerestimation.optimization.authorization import OptimizationExecutionRequest
from zenerestimation.optimization.spaces import ARIMAParameterSpace, LSTMParameterSpace
from zenerestimation.optimization.search_space_m20 import FiniteSearchSpace
from zenerestimation.optimization.search_strategy_m20 import (
    BAYES_TRIAL_LIMIT, BoundedSearchEngine,
)


def request(strategy='grid', family='ARIMA', trials=4, epoch_budget=None):
    return OptimizationExecutionRequest(
        dataset_id='410', development_fingerprint='sha256:'+'a'*64,
        development_cutoff='2023-12-01', model_family=family, strategy=strategy,
        max_trials=trials, max_epoch_folds=epoch_budget,
        validation_protocol_digest='sha256:'+'b'*64,
    )


def arima(n=6):
    return FiniteSearchSpace.from_legacy('ARIMA', ARIMAParameterSpace(p=range(n), d=[1], q=[0]))


def engine(strategy='grid', trials=4, n=6):
    return BoundedSearchEngine(request=request(strategy, trials=trials),
                               parameter_space=arima(n), seed=7, fold_count=3)


def test_mutated_params_rejected_and_private_audit_unchanged():
    e = engine()
    p = e.propose()
    original = e.snapshot()
    p.params['order'] = (999, 999, 999)
    assert e.snapshot() == original
    with pytest.raises(ValueError, match='integrity'):
        e.record_result(p, status='evaluated', score=0.2)
    assert e.snapshot()['pending_sequence'] == 1


def test_fabricated_or_replaced_proposal_rejected():
    e = engine()
    p = e.propose()
    with pytest.raises(ValueError):
        e.record_result(replace(p), status='failed')
    e.record_result(p, status='failed')
    assert e.snapshot()['failed_count'] == 1


def test_mutated_candidate_id_rejected():
    e = engine()
    p = e.propose()
    object.__setattr__(p, 'candidate_id', 'sha256:' + '0'*64)
    with pytest.raises(ValueError, match='integrity'):
        e.record_result(p, status='failed')
    assert e.snapshot()['feedback'] == []


def test_mutated_identity_rejected():
    e = engine()
    p = e.propose()
    object.__setattr__(p, 'sequence', 99)
    with pytest.raises(ValueError, match='integrity'):
        e.record_result(p, status='failed')
    assert e.snapshot()['proposals'][0]['sequence'] == 1


@pytest.mark.parametrize('changes', [
    {'model_family': 'Unknown'},
    {'names': ('bogus', 'd', 'q')},
    {'dimensions': ((0,), (1,), ())},
    {'dimensions': ((0, 0), (1,), (0,))},
    {'dimensions': ((True,), (1,), (0,))},
    {'dimensions': ((-1,), (1,), (0,))},
    {'candidate_count': 999},
    {'candidate_count': True},
])
def test_direct_space_construction_rejects_invalid(changes):
    base = arima(1)
    with pytest.raises((ValueError, TypeError)):
        replace(base, **changes)


def test_neural_space_constructor_requires_epochs():
    with pytest.raises(ValueError):
        FiniteSearchSpace('LSTM', ('window', 'units'), ((4,), (8,)), 1)


def test_failed_attempt_consumes_trial_and_budget_at_boundary():
    space = FiniteSearchSpace.from_legacy('LSTM', LSTMParameterSpace(
        window=[4], units=[8, 16], epochs=[10]))
    e = BoundedSearchEngine(request=request('grid', 'LSTM', 2, 30),
                            parameter_space=space, seed=1, fold_count=3)
    p = e.propose()
    assert p.planned_epoch_folds == 30
    e.record_result(p, status='failed')
    assert e.propose() is None
    assert e.snapshot()['reserved_epoch_folds'] == 30
    assert e.snapshot()['stop_reason'] == 'max_epoch_folds'


def test_invalid_feedback_does_not_release_pending():
    e = engine()
    p = e.propose()
    with pytest.raises(ValueError):
        e.record_result(p, status='evaluated', score=float('inf'))
    with pytest.raises(ValueError):
        e.record_result(p, status='failed', score=0.1)
    assert e.snapshot()['pending_sequence'] == 1
    e.record_result(p, status='failed')


@pytest.mark.parametrize('strategy', ['grid', 'random', 'bayesian'])
def test_exhaustion_no_duplicates(strategy):
    e = engine(strategy, trials=8, n=3)
    indices = []
    while (p := e.propose()) is not None:
        indices.append(p.index)
        e.record_result(p, status='evaluated', score=float(p.index))
    assert len(indices) == len(set(indices)) == 3
    assert e.snapshot()['stop_reason'] == 'space_exhausted'


def test_bayesian_extreme_finite_scores_remain_deterministic():
    def run():
        e = engine('bayesian', trials=4)
        out = []
        for score in (1e308, -1e308, 1e-308, 0.0):
            p = e.propose()
            out.append(p.index)
            e.record_result(p, status='evaluated', score=score)
        assert e.propose() is None
        return out, e.snapshot()
    assert run() == run()


def test_bayesian_trial_ceiling_fails_before_proposals():
    with pytest.raises(ValueError, match='trial limit'):
        engine('bayesian', trials=BAYES_TRIAL_LIMIT + 1)


def test_strategy_family_mismatch_rejected():
    with pytest.raises(ValueError, match='family mismatch'):
        BoundedSearchEngine(request=request(family='Kalman'), parameter_space=arima(),
                            seed=1, fold_count=3)
