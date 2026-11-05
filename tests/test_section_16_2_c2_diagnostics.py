from zenerestimation.optimization.evaluator import ForecastAlignmentError
from zenerestimation.optimization.candidate import CandidateResult
from zenerestimation.optimization.search import GenericOptimizer


def test_alignment_exception_diagnostic_is_compact():
    error = ForecastAlignmentError(
        'fold 2: verified forecast timestamps are required (received positional_only)',
        fold_index=2,
        observed_status='positional_only',
    )
    assert isinstance(error, ValueError)
    assert error.alignment_failure == {
        'reason': 'missing_forecast_dates',
        'fold_index': 2,
        'required_status': 'verified',
        'observed_status': 'positional_only',
    }


def test_failed_candidate_can_retain_safe_alignment_diagnostics():
    error = ForecastAlignmentError('missing dates', fold_index=1, observed_status='positional_only')
    candidate = CandidateResult.failed(
        params={'window': 4}, error=error,
        metadata={'alignment_failure': dict(error.alignment_failure)},
    )
    assert candidate.status == 'failed'
    assert candidate.score is None
    assert candidate.metadata['error_type'] == 'ForecastAlignmentError'
    assert candidate.metadata['alignment_failure']['fold_index'] == 1
    assert 'forecast_dates' not in candidate.metadata['alignment_failure']


def test_best_selection_ignores_failed_candidates():
    good = CandidateResult(params={'window': 4}, score=0.4, metrics={'rmse': 0.4})
    bad = CandidateResult.failed(params={'window': 6}, error=ValueError('bad'))
    assert GenericOptimizer._select_best((bad, good)) is good
