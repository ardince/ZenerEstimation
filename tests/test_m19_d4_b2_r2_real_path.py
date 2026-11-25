"""M19-D4-B2-R2: actual M18 -> D2/D3 -> D4 integration (no mocks)."""
from copy import deepcopy
import json

import pytest

from tests.test_m18_accuracy_profile_ranking import make_result, trusted_context
from zenerestimation.assessment.consolidated_assessment import build_consolidated_assessment
from zenerestimation.assessment.selection_consistency import assess_internal_selection_consistency


def _real(errors):
    context = trusted_context()
    results = {model: make_result(model, context, errors=folds)
               for model, folds in errors.items()}
    return results, context


def _assess(errors):
    results, context = _real(errors)
    return build_consolidated_assessment(results, validation_context=context)


def _assert_boundaries(report):
    assert report['scientific_boundaries'] == {
        'selection_authorized': False,
        'prospective_certification_issued': False,
        'protected_holdout_accessed': False,
        'optimization_executed': False,
    }
    assert report['internal_assessment']['selection_authorized'] is False
    assert report['scientific_eligibility'] is None
    assert report['evidence_relationship']['cross_track_identity_verified'] is False


def test_real_path_unique_winner_and_exact_d3_report():
    errors = {'LSTM': ((0.4, 0.4), (0.6, 0.6)),
              'GRU': ((0.1, 0.1), (0.2, 0.2))}
    results, context = _real(errors)
    report = build_consolidated_assessment(results, validation_context=context)
    upstream = assess_internal_selection_consistency(results, validation_context=context)
    assert report['assessment_version'] == 'm19_d4_b2_consolidated_assessment_v1'
    assert report['status'] == 'available'
    assert report['internal_assessment'] == upstream
    assert upstream['baseline_recommended_model'] == 'GRU'
    assert upstream['recommendation_consistency_fraction'] == 1.0
    _assert_boundaries(report)


def test_real_path_replacement_winner_retains_diagnostics():
    report = _assess({'LSTM': ((0.1, 0.1), (0.8, 0.8)),
                      'GRU': ((0.6, 0.6), (0.4, 0.4))})
    assert report['status'] == 'available'
    assert report['internal_assessment']['baseline_recommended_model'] == 'LSTM'
    assert report['internal_assessment']['recommendation_consistency_fraction'] == 0.5
    _assert_boundaries(report)


def test_real_path_tied_baseline_never_implies_recommendation():
    report = _assess({'LSTM': ((0.2, 0.2), (0.2, 0.2)),
                      'GRU': ((0.2, 0.2), (0.2, 0.2)),
                      'ARIMA': ((0.6, 0.6), (0.6, 0.6))})
    assert report['status'] == 'available'
    assert report['internal_assessment']['baseline_recommended_model'] is None
    assert report['internal_assessment']['recommendation_consistency_fraction'] is None
    _assert_boundaries(report)


def test_real_path_insufficient_models_is_unavailable_not_zero():
    report = _assess({'LSTM': ((0.2, 0.2), (0.6, 0.6))})
    assert report['status'] == 'unavailable_insufficient_models'
    for field in ('recommendation_consistency_fraction',
                  'winner_set_stability_fraction', 'rank_stability_fraction',
                  'recommendation_winner_gap', 'winner_rank_gap'):
        assert report['internal_assessment'][field] is None
    _assert_boundaries(report)


def test_real_path_deterministic_order_and_read_only():
    results, context = _real({'LSTM': ((0.2, 0.2), (0.6, 0.6)),
                              'GRU': ((0.1, 0.1), (0.4, 0.4))})
    before_results = {key: value.to_dict() for key, value in results.items()}
    before_context = deepcopy(context)
    first = build_consolidated_assessment(results, validation_context=context)
    second = build_consolidated_assessment(
        dict(reversed(list(results.items()))), validation_context=context)
    assert first == second
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False)
    assert {key: value.to_dict() for key, value in results.items()} == before_results
    assert context == before_context
    _assert_boundaries(first)


@pytest.mark.parametrize('tamper', ['negative', 'nan', 'calendar', 'alignment', 'profile'])
def test_real_path_rejects_corrupted_m18_evidence(tamper):
    results, context = _real({'LSTM': ((0.2, 0.2), (0.6, 0.6)),
                              'GRU': ((0.1, 0.1), (0.4, 0.4))})
    candidate = results['LSTM'].candidates[0]
    if tamper == 'negative':
        candidate.metadata['fold_metrics'][0]['rmse'] = -1
    elif tamper == 'nan':
        candidate.metadata['fold_metrics'][0]['rmse'] = float('nan')
    elif tamper == 'calendar':
        candidate.metadata['fold_alignment'][0]['forecast_dates'][0] = '2030-03-01'
    elif tamper == 'alignment':
        candidate.metadata['fold_alignment'][0]['status'] = 'positional_only'
    else:
        candidate.metadata['accuracy_profile']['by_date'][0]['predicted_microVolt'] += 1
    with pytest.raises(ValueError):
        build_consolidated_assessment(results, validation_context=context)


def test_real_path_rejects_external_holdout_context():
    results, context = _real({'LSTM': ((0.2, 0.2), (0.6, 0.6)),
                              'GRU': ((0.1, 0.1), (0.4, 0.4))})
    context['scope'] = 'external_holdout'
    with pytest.raises(ValueError):
        build_consolidated_assessment(results, validation_context=context)


def test_real_path_rejects_missing_trusted_context():
    results, _ = _real({'LSTM': ((0.2, 0.2), (0.6, 0.6)),
                        'GRU': ((0.1, 0.1), (0.4, 0.4))})
    with pytest.raises((TypeError, ValueError)):
        build_consolidated_assessment(results, validation_context=None)
