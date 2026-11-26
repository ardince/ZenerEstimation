"""M20-B5-E: simultaneous use of one process-local authorization grant."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from tests.test_m20_b5_e_real_path import make_case, execute


def test_same_grant_two_simultaneous_runs_at_most_one_executes():
    case = make_case()
    barrier = Barrier(3)

    def attempt():
        barrier.wait(timeout=10)
        try:
            return ('outcome', execute(case))
        except PermissionError as exc:
            return ('denied', type(exc).__name__)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(attempt)
        b = pool.submit(attempt)
        barrier.wait(timeout=10)
        results = [a.result(timeout=30), b.result(timeout=30)]

    authorized = [value for kind, value in results if kind == 'outcome']
    denied = [value for kind, value in results if kind == 'denied']
    assert len(authorized) == 1, results
    assert len(denied) == 1, results
    assert authorized[0].status == 'COMPLETED', authorized[0].failure
    # Exactly one successful run of two candidates and two folds.
    assert len(case['recorder']) == 4
    assert len(authorized[0].candidate_results) == 2
    with pytest.raises(PermissionError):
        case['coordinator'].verify(case['grant'], case['request'])
