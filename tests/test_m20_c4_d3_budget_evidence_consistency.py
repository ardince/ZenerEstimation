"""M20-C4-D3: authorized neural/hybrid budget and evidence reconciliation.

Run in the ZenerEstimation repository alongside accepted C4 B/C test modules.
No production modifications; real TensorFlow execution is required.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SOURCES = {
    'LSTM': 'test_m20_c4_b1_lstm_grid_real_path.py',
    'GRU': 'test_m20_c4_b2_gru_grid_real_path.py',
    'LinearTrendLSTM': 'test_m20_c4_c1_linear_trend_lstm_grid_real_path.py',
    'KalmanLSTM': 'test_m20_c4_c2_kalman_lstm_grid_real_path.py',
}


def fixture_module(family):
    path = Path(__file__).resolve().parent / SOURCES[family]
    spec = importlib.util.spec_from_file_location('m20_c4_d3_' + family.lower(), path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def residual_network(model, family):
    return model if family in ('LSTM', 'GRU') else model.residual_model


def check_common_snapshot(case, outcome, expected_statuses):
    snapshot = outcome.search_snapshot
    candidates = outcome.candidate_results
    proposals = snapshot['proposals']
    feedback = snapshot['feedback']
    assert snapshot['request_digest'] == case['request'].digest
    assert snapshot['space_digest'] == case['plan'].search_space_digest
    assert snapshot['strategy'] == 'grid'
    assert snapshot['fold_count'] == 2
    assert snapshot['pending_sequence'] is None
    assert snapshot['proposed_count'] == len(proposals) == len(feedback) == len(candidates)
    assert snapshot['completed_count'] == sum(c.status == 'evaluated' for c in candidates)
    assert snapshot['failed_count'] == sum(c.status == 'failed' for c in candidates)
    assert [c.status for c in candidates] == expected_statuses
    assert snapshot['reserved_epoch_folds'] == sum(p['planned_epoch_folds'] for p in proposals)
    assert snapshot['reserved_epoch_folds'] <= case['request'].max_epoch_folds
    assert snapshot['proposed_count'] <= case['request'].max_trials
    for i, (proposal, result, item) in enumerate(zip(proposals, candidates, feedback), 1):
        assert proposal['sequence'] == item['sequence'] == i
        assert proposal['candidate_id'] == item['candidate_id']
        assert proposal['params'] == result.params
        assert proposal['planned_epoch_folds'] == int(result.params['epochs']) * snapshot['fold_count']
        assert item['status'] == result.status
        if result.status == 'evaluated':
            assert item['score'] == pytest.approx(result.score)
        else:
            assert item['score'] is None
    return snapshot


@pytest.mark.parametrize('family', SOURCES)
def test_successful_trials_reconcile_budget_folds_training_and_provenance(family):
    module = fixture_module(family)
    case = module.make_case(epochs=(2, 3), max_trials=2, max_epoch_folds=10)
    outcome = module.execute(case)
    assert outcome.status == 'COMPLETED', outcome.failure
    assert outcome.failure is None
    assert outcome.request_digest == case['request'].digest
    assert outcome.execution_plan_digest == case['plan'].digest
    snap = check_common_snapshot(case, outcome, ['evaluated', 'evaluated'])
    assert snap['reserved_epoch_folds'] == 10
    assert [p['planned_epoch_folds'] for p in snap['proposals']] == [4, 6]
    assert len(case['created_models']) == 4
    assert len({id(m) for m in case['created_models']}) == 4

    # Evaluator creates a fresh fold model for each candidate, in proposal order.
    for candidate_index, candidate in enumerate(outcome.candidate_results):
        assert candidate.metadata['internal_provenance'] == case['context']
        assert candidate.metadata['selection_metric'] == 'rmse'
        metrics = candidate.metadata['fold_metrics']
        alignments = candidate.metadata['fold_alignment']
        assert len(metrics) == len(alignments) == 2
        scores = [float(m['rmse']) for m in metrics]
        assert np.isfinite(scores).all()
        assert candidate.score == pytest.approx(sum(scores) / len(scores))
        for alignment, expected in zip(alignments, case['context']['fold_validation_dates']):
            assert alignment['status'] == 'verified'
            assert alignment['timestamp_match'] is True
            assert pd.DatetimeIndex(alignment['forecast_dates']).equals(pd.DatetimeIndex(expected))
            assert pd.DatetimeIndex(alignment['validation_dates']).equals(pd.DatetimeIndex(expected))

        actual_epochs = []
        for model in case['created_models'][2*candidate_index:2*candidate_index+2]:
            neural = residual_network(model, family)
            assert neural.training_recorder is not None
            assert neural.history is not None
            losses = neural.history.history['loss']
            assert losses and np.isfinite(np.asarray(losses, dtype=float)).all()
            actual_epochs.append(len(losses))
            assert 0 < len(losses) <= int(candidate.params['epochs'])
            assert pd.to_datetime(model.dataset.data['ds']).max() < pd.Timestamp(
                case['declaration'].forecast_dates[0]
            )
        print(f'{family}: candidate={candidate_index + 1}, '
              f'reserved={snap["proposals"][candidate_index]["planned_epoch_folds"]}, '
              f'observed_training_epochs={actual_epochs}, ranking_rmse={candidate.score:.12g}')

    best = min(outcome.candidate_results, key=lambda c: c.score)
    assert outcome.optimization_result.best_params == best.params
    assert outcome.optimization_result.best_score == pytest.approx(best.score)


@pytest.mark.parametrize('family', SOURCES)
def test_failed_trial_keeps_reserved_budget_and_failure_evidence(family):
    module = fixture_module(family)
    case = module.make_case(development_length=10, window=8,
                            epochs=(2,), max_trials=1, max_epoch_folds=4)
    outcome = module.execute(case)
    assert outcome.status == 'ALL_FAILED', outcome.failure
    assert outcome.failure is None
    assert outcome.optimization_result is None
    snap = check_common_snapshot(case, outcome, ['failed'])
    assert snap['reserved_epoch_folds'] == 4
    assert snap['completed_count'] == 0
    assert snap['failed_count'] == 1
    failed = outcome.candidate_results[0]
    assert failed.score is None
    assert not failed.metrics
    # Reserved budget is not a claim that four epochs actually ran.
    print(f'{family}: audited failure retained reserved_epoch_folds=4')
