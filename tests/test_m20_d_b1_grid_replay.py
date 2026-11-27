"""M20-D-B1: independent replay of authorized real-path Grid Search evidence."""
from copy import deepcopy
from dataclasses import replace

import pytest

from tests.test_m20_b5_e_real_path import make_case, execute
from zenerestimation.optimization.search_space_m20 import FiniteSearchSpace
from zenerestimation.optimization.search_replay_m20 import (
    SearchReplayVerificationError, verify_grid_search_replay,
)

def _space():
    return FiniteSearchSpace('ARIMA', ('p', 'd', 'q'), ((0, 1), (0,), (0,)), 2)


def _verify(case, outcome):
    return verify_grid_search_replay(
        request=case['request'], plan=case['plan'],
        parameter_space=_space(), outcome=outcome,
    )


def _tampered(outcome, change):
    snapshot = deepcopy(outcome.search_snapshot)
    change(snapshot)
    return replace(outcome, search_snapshot=snapshot)


def test_completed_real_path_verified_without_training_again():
    case = make_case()
    outcome = execute(case)
    before = deepcopy(outcome.search_snapshot)
    fits = len(case['recorder'])
    report = _verify(case, outcome)
    assert report['status'] == 'VERIFIED'
    assert report['verified_proposals'] == 2
    assert report['verified_feedback'] == 2
    assert report['verified_reserved_epoch_folds'] == 0
    assert report['outcome_status'] == 'COMPLETED'
    assert len(case['recorder']) == fits
    assert outcome.search_snapshot == before


def test_aborted_pending_real_path_is_valid():
    case = make_case(shift_dates=True)
    outcome = execute(case)
    assert outcome.status == 'ABORTED'
    report = _verify(case, outcome)
    assert report['verified_proposals'] == 1
    assert report['verified_feedback'] == 0
    assert report['outcome_status'] == 'ABORTED'


@pytest.mark.parametrize('change', [
    lambda s: s['proposals'][0].__setitem__('index', 1),
    lambda s: s['proposals'][0].__setitem__('sequence', 2),
    lambda s: s['proposals'][0].__setitem__('candidate_id', 'sha256:' + '0' * 64),
    lambda s: s['proposals'][0]['params'].__setitem__('order', (9, 0, 0)),
    lambda s: s['proposals'].reverse(),
    lambda s: s['proposals'].append(deepcopy(s['proposals'][0])),
    lambda s: s['feedback'][0].__setitem__('score', 999.0),
    lambda s: s['feedback'][0].__setitem__('status', 'failed'),
    lambda s: s['feedback'].reverse(),
    lambda s: s.__setitem__('reserved_epoch_folds', 1),
    lambda s: s.__setitem__('proposed_count', 1),
    lambda s: s.__setitem__('completed_count', 1),
    lambda s: s.__setitem__('stop_reason', 'space_exhausted'),
    lambda s: s.__setitem__('seed', 8),
    lambda s: s.__setitem__('pending_sequence', 1),
])
def test_snapshot_tampering_rejected(change):
    case = make_case()
    outcome = execute(case)
    with pytest.raises(SearchReplayVerificationError):
        _verify(case, _tampered(outcome, change))


def test_request_plan_space_mismatch_rejected():
    case = make_case()
    outcome = execute(case)
    wrong_space = FiniteSearchSpace('ARIMA', ('p', 'd', 'q'), ((1, 0), (0,), (0,)), 2)
    with pytest.raises(SearchReplayVerificationError):
        verify_grid_search_replay(request=case['request'], plan=case['plan'],
                                  parameter_space=wrong_space, outcome=outcome)
    with pytest.raises(SearchReplayVerificationError):
        verify_grid_search_replay(request=case['request'],
                                  plan=replace(case['plan'], seed=8),
                                  parameter_space=_space(), outcome=outcome)


def test_candidate_result_mismatch_rejected():
    case = make_case()
    outcome = execute(case)
    changed = deepcopy(outcome.candidate_results[0])
    changed.params['order'] = (9, 0, 0)
    with pytest.raises(SearchReplayVerificationError):
        _verify(case, replace(outcome, candidate_results=(changed, outcome.candidate_results[1])))


def test_missing_feedback_rejected_for_completed_outcome():
    case = make_case()
    outcome = execute(case)
    with pytest.raises(SearchReplayVerificationError):
        _verify(case, _tampered(outcome, lambda s: s['feedback'].pop()))


def test_unsupported_snapshot_version_rejected():
    case = make_case()
    outcome = execute(case)

    altered = _tampered(
        outcome,
        lambda snapshot: snapshot.__setitem__(
            'version', 'unsupported-search-version'
        ),
    )

    with pytest.raises(SearchReplayVerificationError):
        _verify(case, altered)