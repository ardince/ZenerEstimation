from copy import deepcopy
from dataclasses import replace

import pytest

from tests.test_m20_c2_b_arima_random_real_path import execute, make_case
from zenerestimation.optimization.search_replay_m20 import (
    SearchReplayVerificationError,
    verify_grid_search_replay,
    verify_random_search_replay,
)


def _replay_inputs(case, outcome):
    return {
        'request': case['request'],
        'plan': case['plan'],
        'parameter_space': case['space'],
        'outcome': outcome,
    }


@pytest.mark.parametrize(
    ('max_trials', 'stop_reason'),
    [(1, 'max_trials'), (2, 'max_trials'), (4, 'space_exhausted')],
)
def test_random_replay_verifies_real_arima_outcome(max_trials, stop_reason):
    case = make_case(seed=7, max_trials=max_trials)
    outcome = execute(case)
    before = deepcopy(outcome.search_snapshot)

    report = verify_random_search_replay(**_replay_inputs(case, outcome))

    expected_trials = min(max_trials, case['space'].candidate_count)
    assert before['stop_reason'] == stop_reason
    assert report['status'] == 'VERIFIED'
    assert report['strategy'] == 'random'
    assert report['outcome_status'] == 'COMPLETED'
    assert report['verified_proposals'] == expected_trials
    assert report['verified_feedback'] == expected_trials
    assert report['verified_reserved_epoch_folds'] == 0
    assert outcome.search_snapshot == before


def test_random_replay_rejects_reordered_history():
    case = make_case(seed=7, max_trials=2)
    outcome = execute(case)
    snapshot = deepcopy(outcome.search_snapshot)

    for key in ('proposals', 'feedback'):
        snapshot[key].reverse()
        for sequence, item in enumerate(snapshot[key], start=1):
            item['sequence'] = sequence

    changed = replace(
        outcome,
        search_snapshot=snapshot,
        candidate_results=outcome.candidate_results[::-1],
    )
    with pytest.raises(
        SearchReplayVerificationError, match='Random Search candidate index mismatch'
    ):
        verify_random_search_replay(**_replay_inputs(case, changed))


def test_grid_replay_rejects_random_outcome():
    case = make_case()
    outcome = execute(case)

    with pytest.raises(SearchReplayVerificationError, match='not a Grid Search'):
        verify_grid_search_replay(**_replay_inputs(case, outcome))


# M20-D-B2 extended acceptance: independent Kalman and integrity coverage.
from tests.test_m20_c2_c_kalman_random_real_path import (
    execute as execute_kalman,
    make_case as make_kalman_case,
)


def _alter_snapshot(outcome, change):
    snapshot = deepcopy(outcome.search_snapshot)
    change(snapshot)
    return replace(outcome, search_snapshot=snapshot)


@pytest.mark.parametrize('max_trials', [1, 2, 4])
def test_random_replay_verifies_real_kalman_outcome(max_trials):
    case = make_kalman_case(seed=11, max_trials=max_trials)
    outcome = execute_kalman(case)
    before = deepcopy(outcome.search_snapshot)
    report = verify_random_search_replay(**_replay_inputs(case, outcome))
    count = min(max_trials, case['space'].candidate_count)
    assert outcome.status == 'COMPLETED'
    assert report['status'] == 'VERIFIED'
    assert report['strategy'] == 'random'
    assert report['verified_proposals'] == count
    assert report['verified_feedback'] == count
    assert report['outcome_status'] == 'COMPLETED'
    assert outcome.search_snapshot == before


def test_random_replay_verifies_real_kalman_all_failed():
    case = make_kalman_case(short_history=True)
    outcome = execute_kalman(case)
    assert outcome.status == 'ALL_FAILED'
    report = verify_random_search_replay(**_replay_inputs(case, outcome))
    assert report['status'] == 'VERIFIED'
    assert report['outcome_status'] == 'ALL_FAILED'
    assert report['verified_proposals'] == 2
    assert report['verified_feedback'] == 2


@pytest.mark.parametrize('change', [
    lambda s: s.__setitem__('seed', s['seed'] + 1),
    lambda s: s['proposals'][0].__setitem__('index', -1),
    lambda s: s['proposals'][0].__setitem__('candidate_id', 'sha256:' + '0' * 64),
    lambda s: s['proposals'][0].__setitem__('params', {'order': (99, 0, 0)}),
    lambda s: s['feedback'][0].__setitem__('score', 999999.0),
    lambda s: s['feedback'][0].__setitem__('status', 'failed'),
    lambda s: s.__setitem__('proposed_count', 0),
    lambda s: s.__setitem__('completed_count', 0),
    lambda s: s.__setitem__('stop_reason', 'space_exhausted'),
    lambda s: s.__setitem__('version', 'unsupported-search-version'),
])
def test_random_replay_rejects_tampered_snapshot(change):
    case = make_case(seed=7, max_trials=2)
    outcome = execute(case)
    with pytest.raises(SearchReplayVerificationError):
        verify_random_search_replay(**_replay_inputs(case, _alter_snapshot(outcome, change)))


def test_random_replay_rejects_candidate_result_tampering():
    case = make_case(seed=7, max_trials=2)
    outcome = execute(case)
    changed = deepcopy(outcome.candidate_results[0])
    changed.params['order'] = (99, 0, 0)
    altered = replace(outcome, candidate_results=(changed, *outcome.candidate_results[1:]))
    with pytest.raises(SearchReplayVerificationError):
        verify_random_search_replay(**_replay_inputs(case, altered))


def test_random_replay_rejects_changed_approved_seed():
    case = make_case(seed=7, max_trials=2)
    outcome = execute(case)
    inputs = _replay_inputs(case, outcome)
    inputs['plan'] = replace(case['plan'], seed=8)
    with pytest.raises(SearchReplayVerificationError):
        verify_random_search_replay(**inputs)
