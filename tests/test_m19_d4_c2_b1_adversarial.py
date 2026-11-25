"""M19-D4-C2-B1: adversarial verification of existing assessment boundaries.

No protected holdout observations, new authorization, or production changes.
Independent fixtures do not establish cross-track provenance equivalence.
"""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from tests.test_m18_accuracy_profile_ranking import make_result, trusted_context
from tests.test_m19_b2_i2_c1_alignment import case as c1_case  # transitive fixture
from tests.test_m19_b2_i2_c3_evidence_integration import integrated_args, temporal_for
from tests.test_m19_d1_b2_b2_authorization_integration import (
    _internal as d1_internal, _c2_inputs as d1_c2_inputs,
    _decision as d1_decision,
)
from zenerestimation.assessment.consolidated_assessment import build_consolidated_assessment
from zenerestimation.assessment.evidence_integration import (
    TemporalSummaryStatus, integrate_assessment_evidence,
)
from zenerestimation.assessment.eligibility import VerificationStatus
from zenerestimation.comparison.generalization_gap_assessment import assess_generalization_gap

_DENIED = (
    'protected_holdout_assessment_permitted',
    'protected_holdout_comparison_permitted',
    'protected_holdout_ranking_permitted',
    'protected_holdout_disclosure_permitted',
)


def _internal_inputs():
    trusted = trusted_context()
    results = {
        'LSTM': make_result('LSTM', trusted, errors=((0.4, 0.4), (0.6, 0.6))),
        'GRU': make_result('GRU', trusted, errors=((0.1, 0.1), (0.2, 0.2))),
    }
    return results, trusted


def _d4():
    results, trusted = _internal_inputs()
    return build_consolidated_assessment(results, validation_context=trusted)


def _d1(**overrides):
    args = {**d1_internal(), 'benchmark_authorization_inputs': d1_c2_inputs()}
    args.update(overrides)
    return assess_generalization_gap(**args)


def _assert_denied(report):
    assert report['status'] in ('AUTHORIZATION_REQUIRED', 'AUTHORIZATION_REJECTED')
    assert report['holdout_assessed'] is False
    assert report['comparability_verified'] is False
    assert report['generalization_gap'] is None
    assert report['generalization_gap_metric'] is None
    for key in _DENIED:
        assert report[key] is False


# A — M18 internal evidence integrity: real upstream validators.
def test_a01_negative_fold_rmse_rejected():
    results, trusted = _internal_inputs()
    results['LSTM'].candidates[0].metadata['fold_metrics'][0]['rmse'] = -1
    with pytest.raises(ValueError):
        build_consolidated_assessment(results, validation_context=trusted)


def test_a02_nonfinite_fold_rmse_rejected():
    results, trusted = _internal_inputs()
    results['LSTM'].candidates[0].metadata['fold_metrics'][0]['rmse'] = float('nan')
    with pytest.raises(ValueError):
        build_consolidated_assessment(results, validation_context=trusted)


def test_a03_forecast_calendar_tamper_rejected():
    results, trusted = _internal_inputs()
    results['LSTM'].candidates[0].metadata['fold_alignment'][0]['forecast_dates'][0] = '2030-03-01'
    with pytest.raises(ValueError):
        build_consolidated_assessment(results, validation_context=trusted)


def test_a04_accuracy_profile_tamper_rejected():
    results, trusted = _internal_inputs()
    results['LSTM'].candidates[0].metadata['accuracy_profile']['by_date'][0]['predicted_microVolt'] += 1
    with pytest.raises(ValueError):
        build_consolidated_assessment(results, validation_context=trusted)


# B — observed evidence integrity: real integration recomputation.
def test_b01_declaration_digest_mismatch_rejected(integrated_args):
    args = dict(integrated_args)
    args['metric_result'] = replace(args['metric_result'], declaration_digest='wrong-digest')
    with pytest.raises(ValueError, match='declaration_digest'):
        integrate_assessment_evidence(**args)



def test_b02_coverage_mask_mismatch_rejected(integrated_args):
    args = dict(integrated_args)
    alignment = args["alignment_result"]

    # Select an eligible row from the existing valid alignment.
    eligible_index = next(
        i for i, row in enumerate(alignment.rows)
        if row.eligible
    )

    original_row = alignment.rows[eligible_index]

    # Construct a valid excluded row.
    # The original observability report still identifies
    # this observation as verified and eligible.
    tampered_row = replace(
        original_row,
        eligible=False,
        observed_microVolt=None,
        exclusion_reasons=("PROVENANCE_NOT_VERIFIED",),
    )

    tampered_rows = list(alignment.rows)
    tampered_rows[eligible_index] = tampered_row

    args["alignment_result"] = replace(
        alignment,
        rows=tuple(tampered_rows),
    )

    # The alignment and observability coverage now disagree.
    assert (
        args["alignment_result"].coverage_mask
        != alignment.coverage_mask
    )

    # The integration contract must reject the contradiction.
    with pytest.raises(ValueError):
        integrate_assessment_evidence(**args)



def test_b03_forged_observed_metrics_rejected(integrated_args):
    args = dict(integrated_args)
    metrics = args['metric_result']
    assert metrics.rmse_microVolt is not None
    args['metric_result'] = replace(metrics, rmse_microVolt=metrics.rmse_microVolt + 1)
    with pytest.raises(ValueError, match='metric'):
        integrate_assessment_evidence(**args)


def test_b04_temporal_incomplete_preserved_with_partial_accuracy(integrated_args):
    args = dict(integrated_args)
    args['temporal_evidence'] = temporal_for(
        args['declaration'], {'chronology': VerificationStatus.UNKNOWN}
    )
    report = integrate_assessment_evidence(**args)
    assert report.temporal_status is TemporalSummaryStatus.INCOMPLETE
    assert report.coverage_status == 'partial'
    assert report.score_available is True
    assert 'scientific_classification' not in report.to_dict()


# C — authorization: use real C2 path unless injecting contradictory downstream output.
def test_c01_supplied_authorization_disagreement_rejected():
    report = _d1(supplied_authorization_decision=d1_decision('AUTHORIZATION_REQUIRED'))
    _assert_denied(report)
    assert report['status'] == 'AUTHORIZATION_REJECTED'
    assert 'SUPPLIED_AUTHORIZATION_DECISION_MISMATCH' in report['reasons']


def test_c02_forged_positive_permission_rejected_by_facade(monkeypatch):
    baseline = _d1()
    forged = deepcopy(baseline)
    forged['protected_holdout_assessment_permitted'] = True
    monkeypatch.setattr(
        'zenerestimation.comparison.generalization_gap_assessment.'
        'reconcile_generalization_authorization',
        lambda **kwargs: deepcopy(forged),
    )
    with pytest.raises(ValueError, match='fail-closed'):
        _d1()


def test_c03_required_authorization_never_becomes_approval(monkeypatch):
    # A controlled synthetic upstream decision tests downstream interpretation only.
    decision = d1_decision('AUTHORIZATION_REQUIRED')
    monkeypatch.setattr(
        'zenerestimation.comparison.generalization_authorization_integration.'
        'assess_independent_benchmark_authorization',
        lambda **kwargs: deepcopy(decision),
    )
    report = _d1(supplied_authorization_decision=deepcopy(decision))
    _assert_denied(report)
    assert report['status'] == 'AUTHORIZATION_REQUIRED'


def test_c04_benchmark_identity_does_not_prove_comparability():
    report = _d1()
    _assert_denied(report)
    assert isinstance(report['selected_model'], str)
    assert report['selected_model']
    assert report['comparability_verified'] is False


# D — interpretive boundaries.
def test_d01_stable_ranking_is_not_selection_authorization():
    report = _d4()
    assert report['status'] == 'available'
    assert report['internal_assessment']['baseline_recommended_model'] == 'GRU'
    assert report['internal_assessment']['selection_authorized'] is False
    assert report['scientific_boundaries']['selection_authorized'] is False


def test_d02_matching_labels_cannot_authenticate_cross_track_identity():
    report = _d4()
    relation = report['evidence_relationship']
    assert relation['cross_track_identity_verified'] is False
    assert relation['dataset_id_match'] is None  # no eligibility track supplied
    assert report['scientific_boundaries']['prospective_certification_issued'] is False


def test_d03_insufficient_ranking_is_unavailable_not_zero():
    trusted = trusted_context()
    results = {'LSTM': make_result('LSTM', trusted, errors=((0.2, 0.2), (0.6, 0.6)))}
    report = build_consolidated_assessment(results, validation_context=trusted)
    assert report['status'] == 'unavailable_insufficient_models'
    for field in ('recommendation_consistency_fraction', 'winner_set_stability_fraction',
                  'rank_stability_fraction', 'recommendation_winner_gap', 'winner_rank_gap'):
        assert report['internal_assessment'][field] is None
    assert report['scientific_boundaries']['selection_authorized'] is False


# E — snapshots are detached and computations reproducible.
def test_e01_snapshot_mutation_does_not_mutate_sources(integrated_args):
    original = {k: v.to_dict() for k, v in integrated_args.items() if hasattr(v, 'to_dict')}
    first = integrate_assessment_evidence(**integrated_args).to_dict()
    second = integrate_assessment_evidence(**integrated_args).to_dict()
    assert first == second
    first['evidence']['observability']['coverage_mask'][0] = False
    assert second['evidence']['observability']['coverage_mask'][0] is True
    assert {k: v.to_dict() for k, v in integrated_args.items() if hasattr(v, 'to_dict')} == original
    d4a, d4b = _d4(), _d4()
    d1a, d1b = _d1(), _d1()
    for left, right in ((d4a, d4b), (d1a, d1b)):
        assert json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False
        )
