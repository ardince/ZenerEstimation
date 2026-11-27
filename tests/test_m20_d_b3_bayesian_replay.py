"""M20-D-B3: independent Bayesian replay on authorized real-model outcomes."""
from copy import deepcopy
from dataclasses import replace

import pytest

from tests.test_m20_c3_b_arima_bayesian_real_path import make_case as arima_case, execute as arima_execute
from tests.test_m20_c3_c_kalman_bayesian_real_path import make_case as kalman_case, execute as kalman_execute
from zenerestimation.optimization.search_replay_m20 import (
    SearchReplayVerificationError, verify_bayesian_search_replay,
    verify_grid_search_replay, verify_random_search_replay,
)


def verify(case, outcome):
    return verify_bayesian_search_replay(
        request=case['request'], plan=case['plan'],
        parameter_space=case['space'], outcome=outcome)


@pytest.mark.parametrize('factory,execute', [(arima_case, arima_execute),
                                             (kalman_case, kalman_execute)])
@pytest.mark.parametrize('trials', [1, 3, 5])
def test_real_bayesian_replay(factory, execute, trials):
    case = factory(max_trials=trials)
    outcome = execute(case)
    before = deepcopy(outcome.search_snapshot)
    report = verify(case, outcome)
    assert report['status'] == 'VERIFIED'
    assert report['strategy'] == 'bayesian'
    assert report['verified_proposals'] == min(trials, 4)
    assert outcome.search_snapshot == before


def test_all_failed_kalman_replays():
    case = kalman_case(short_history=True)
    outcome = kalman_execute(case)
    assert outcome.status == 'ALL_FAILED'
    assert verify(case, outcome)['outcome_status'] == 'ALL_FAILED'


@pytest.mark.parametrize('change', [
    lambda s: s['proposals'][0].__setitem__('index', (s['proposals'][0]['index'] + 1) % 4),
    lambda s: s['proposals'][1].__setitem__('index', (s['proposals'][1]['index'] + 1) % 4),
    lambda s: s['proposals'][1].__setitem__('candidate_id', 'sha256:' + '0' * 64),
    lambda s: s['feedback'][0].__setitem__('score', 999.0),
    lambda s: s.__setitem__('seed', 99),
    lambda s: s.__setitem__('stop_reason', 'space_exhausted'),
    lambda s: s.__setitem__('completed_count', 0),
    lambda s: s['proposals'].reverse(),
])
def test_tampered_bayesian_evidence_rejected(change):
    case = arima_case()
    outcome = arima_execute(case)
    snapshot = deepcopy(outcome.search_snapshot)
    change(snapshot)
    with pytest.raises(SearchReplayVerificationError):
        verify(case, replace(outcome, search_snapshot=snapshot))


def test_wrong_seed_and_strategy_rejected():
    case = arima_case()
    outcome = arima_execute(case)
    with pytest.raises(SearchReplayVerificationError):
        verify_bayesian_search_replay(request=case['request'],
            plan=replace(case['plan'], seed=99), parameter_space=case['space'], outcome=outcome)
    for verifier in (verify_grid_search_replay, verify_random_search_replay):
        with pytest.raises(SearchReplayVerificationError):
            verifier(request=case['request'], plan=case['plan'],
                     parameter_space=case['space'], outcome=outcome)
