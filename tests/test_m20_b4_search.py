"""Focused M20-B4 proposal-only tests: no model training or data access."""
import pytest

from zenerestimation.optimization.authorization import OptimizationExecutionRequest
from zenerestimation.optimization.spaces import (
    ARIMAParameterSpace, LSTMParameterSpace, KalmanParameterSpace)
from zenerestimation.optimization.search_space_m20 import (
    FiniteSearchSpace, candidate_digest)
from zenerestimation.optimization.search_strategy_m20 import BoundedSearchEngine


def request(strategy='grid', family='ARIMA', trials=3, epoch_budget=None):
    return OptimizationExecutionRequest(
        dataset_id='410', development_fingerprint='sha256:'+'a'*64,
        development_cutoff='2023-12-01', model_family=family, strategy=strategy,
        max_trials=trials, max_epoch_folds=epoch_budget,
        validation_protocol_digest='sha256:'+'b'*64)


def arima():
    return FiniteSearchSpace.from_legacy('ARIMA', ARIMAParameterSpace(p=[0, 1, 2], d=[1], q=[0, 1]))


def run(strategy, seed=5):
    engine = BoundedSearchEngine(request=request(strategy), parameter_space=arima(),
                                 seed=seed, fold_count=3)
    indices = []
    while (proposal := engine.propose()) is not None:
        indices.append(proposal.index)
        engine.record_result(proposal, status='evaluated', score=float(proposal.index + 1))
    return indices, engine.snapshot()


@pytest.mark.parametrize('strategy', ['grid', 'random', 'bayesian'])
def test_replay(strategy):
    assert run(strategy) == run(strategy)


def test_grid_order_and_trial_limit():
    indices, snapshot = run('grid')
    assert indices == [0, 1, 2]
    assert snapshot['stop_reason'] == 'max_trials'


def test_random_unique_and_seed_sensitive():
    a, _ = run('random', 5)
    b, _ = run('random', 9)
    assert len(set(a)) == 3
    assert a != b


def test_candidate_digest_order_independent_and_type_sensitive():
    assert candidate_digest({'a': 1, 'b': (1, 2)}) == candidate_digest({'b': (1, 2), 'a': 1})
    assert candidate_digest({'a': 1}) != candidate_digest({'a': 1.0})
    assert candidate_digest({'a': 1}) != candidate_digest({'a': True})


def test_space_rejects_duplicate_dimension_values():
    with pytest.raises(ValueError, match='duplicate'):
        FiniteSearchSpace.from_legacy('ARIMA', ARIMAParameterSpace(p=[1, 1], d=[0], q=[0]))


def test_space_rejects_wrong_family():
    with pytest.raises(TypeError):
        FiniteSearchSpace.from_legacy('Kalman', ARIMAParameterSpace(p=[1], d=[0], q=[0]))


def test_kalman_schema():
    space = FiniteSearchSpace.from_legacy('Kalman', KalmanParameterSpace(
        process_noise=[0.01], drift_noise=[0.001], regime_factor=[2.0], regime_multiplier=[10.0]))
    assert set(space.candidate_at(0)) == {'process_noise', 'drift_noise', 'regime_factor', 'regime_multiplier'}


def test_neural_cost_and_failed_attempt_consumes_budget():
    space = FiniteSearchSpace.from_legacy('LSTM', LSTMParameterSpace(
        window=[4], units=[8, 16], epochs=[10], batch_size=[8]))
    engine = BoundedSearchEngine(request=request('grid', 'LSTM', 3, 30),
                                 parameter_space=space, seed=0, fold_count=3)
    proposal = engine.propose()
    assert proposal.planned_epoch_folds == 30
    engine.record_result(proposal, status='failed')
    assert engine.propose() is None
    assert engine.snapshot()['reserved_epoch_folds'] == 30
    assert engine.snapshot()['failed_count'] == 1


def test_neural_first_candidate_over_budget():
    space = FiniteSearchSpace.from_legacy('LSTM', LSTMParameterSpace(
        window=[4], units=[8], epochs=[10]))
    engine = BoundedSearchEngine(request=request('grid', 'LSTM', 2, 29),
                                 parameter_space=space, seed=0, fold_count=3)
    assert engine.propose() is None
    assert engine.snapshot()['stop_reason'] == 'max_epoch_folds'


def test_missing_neural_epochs_rejected():
    with pytest.raises(ValueError, match='epochs'):
        FiniteSearchSpace.from_legacy('LSTM', LSTMParameterSpace(window=[4], units=[8]))


def test_pending_and_duplicate_feedback_rejected():
    engine = BoundedSearchEngine(request=request(), parameter_space=arima(), seed=1, fold_count=3)
    p = engine.propose()
    with pytest.raises(RuntimeError):
        engine.propose()
    with pytest.raises(ValueError):
        engine.record_result(p, status='evaluated', score=float('nan'))
    engine.record_result(p, status='failed')
    with pytest.raises(ValueError):
        engine.record_result(p, status='failed')


def test_all_failed_search_is_auditable():
    engine = BoundedSearchEngine(request=request(trials=2), parameter_space=arima(), seed=1, fold_count=3)
    for _ in range(2):
        p = engine.propose()
        engine.record_result(p, status='failed')
    assert engine.propose() is None
    assert engine.snapshot()['failed_count'] == 2
    assert engine.snapshot()['completed_count'] == 0


def test_bayesian_pool_limit():
    large = FiniteSearchSpace.from_legacy('ARIMA', ARIMAParameterSpace(
        p=range(5000), d=[0], q=[0]))
    with pytest.raises(ValueError, match='pool'):
        BoundedSearchEngine(request=request('bayesian'), parameter_space=large,
                            seed=0, fold_count=3)


def test_snapshot_is_detached():
    engine = BoundedSearchEngine(request=request(), parameter_space=arima(), seed=1, fold_count=3)
    p = engine.propose()
    snapshot = engine.snapshot()
    snapshot['proposals'][0]['params']['order'] = (99, 99, 99)
    assert engine.snapshot()['proposals'][0]['params'] == p.params


def test_reject_bool_seed_and_fold_count():
    for kwargs in ({'seed': True, 'fold_count': 3}, {'seed': 1, 'fold_count': True}):
        with pytest.raises(ValueError):
            BoundedSearchEngine(request=request(), parameter_space=arima(), **kwargs)
