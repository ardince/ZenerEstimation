"""M20-C4-D2: reproducibility of four real neural/hybrid authorized paths.

Uses accepted C4-B/C fixtures; each replay has a new runner and grant.
Numerical comparisons are within-environment, not cross-platform guarantees.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_TEST_DIR = Path(__file__).resolve().parent
_SOURCES = {
    'LSTM': 'test_m20_c4_b1_lstm_grid_real_path.py',
    'GRU': 'test_m20_c4_b2_gru_grid_real_path.py',
    'LinearTrendLSTM': 'test_m20_c4_c1_linear_trend_lstm_grid_real_path.py',
    'KalmanLSTM': 'test_m20_c4_c2_kalman_lstm_grid_real_path.py',
}
# Tolerance deliberately matches the established holdout-invariance tests.
REL_TOL = 1e-5
ABS_TOL = 1e-5


def _fixture(family):
    path = _TEST_DIR / _SOURCES[family]
    spec = importlib.util.spec_from_file_location('m20_c4_d2_' + family.lower(), path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_replay(module):
    case = module.make_case(epochs=(2,), max_trials=1, max_epoch_folds=4)
    outcome = module.execute(case)
    assert outcome.status == 'COMPLETED', outcome.failure
    assert outcome.failure is None
    assert len(outcome.candidate_results) == 1
    return case, outcome


@pytest.mark.parametrize('family', list(_SOURCES))
def test_authorized_neural_family_replay_is_structurally_and_numerically_consistent(family):
    module = _fixture(family)
    case_a, result_a = _run_replay(module)
    case_b, result_b = _run_replay(module)

    # Independent authorization and factory instances; equivalent declarations.
    assert case_a['runner'] is not case_b['runner']
    assert case_a['coordinator'] is not case_b['coordinator']
    assert case_a['created_models'] is not case_b['created_models']
    assert case_a['context'] == case_b['context']
    assert case_a['request'].digest == case_b['request'].digest
    assert case_a['plan'].digest == case_b['plan'].digest
    assert result_a.request_digest == result_b.request_digest
    assert result_a.execution_plan_digest == result_b.execution_plan_digest

    a, b = result_a.candidate_results[0], result_b.candidate_results[0]
    assert a.status == b.status == 'evaluated'
    assert a.params == b.params
    assert a.metadata['internal_provenance'] == b.metadata['internal_provenance']
    assert a.metadata['selection_metric'] == b.metadata['selection_metric'] == 'rmse'

    folds_a, folds_b = a.metadata['fold_metrics'], b.metadata['fold_metrics']
    assert len(folds_a) == len(folds_b) == 2
    scores_a = [float(f['rmse']) for f in folds_a]
    scores_b = [float(f['rmse']) for f in folds_b]
    assert np.isfinite(scores_a).all() and np.isfinite(scores_b).all()
    np.testing.assert_allclose(scores_a, scores_b, rtol=REL_TOL, atol=ABS_TOL,
                               err_msg=f'{family}: fold RMSE replay mismatch')
    assert a.score == pytest.approx(sum(scores_a) / 2)
    assert b.score == pytest.approx(sum(scores_b) / 2)
    assert a.score == pytest.approx(b.score, rel=REL_TOL, abs=ABS_TOL)
    assert result_a.optimization_result.best_params == result_b.optimization_result.best_params
    assert result_a.optimization_result.best_score == pytest.approx(
        result_b.optimization_result.best_score, rel=REL_TOL, abs=ABS_TOL)

    alignment_a = a.metadata['fold_alignment']
    alignment_b = b.metadata['fold_alignment']
    assert len(alignment_a) == len(alignment_b) == 2
    for left, right, expected in zip(
        alignment_a, alignment_b, case_a['context']['fold_validation_dates']
    ):
        assert left['status'] == right['status'] == 'verified'
        assert left['timestamp_match'] is right['timestamp_match'] is True
        for key in ('forecast_dates', 'validation_dates'):
            assert pd.DatetimeIndex(left[key]).equals(pd.DatetimeIndex(expected))
            assert pd.DatetimeIndex(right[key]).equals(pd.DatetimeIndex(expected))

    for result in (result_a, result_b):
        snap = result.search_snapshot
        assert snap['proposed_count'] == 1
        assert snap['completed_count'] == 1
        assert snap['failed_count'] == 0
        assert snap['reserved_epoch_folds'] == 4
        assert [p['planned_epoch_folds'] for p in snap['proposals']] == [4]
        assert snap['pending_sequence'] is None
    assert result_a.search_snapshot['stop_reason'] == result_b.search_snapshot['stop_reason']
    print(f'\n{family}: score_a={a.score:.12g}, score_b={b.score:.12g}, '
          f'abs_delta={abs(a.score-b.score):.6g}, '
          f'max_fold_rmse_delta={max(abs(x-y) for x,y in zip(scores_a,scores_b)):.6g}')
