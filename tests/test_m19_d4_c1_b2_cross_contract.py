"""M19-D4-C1-B2: read-only cross-contract boundary verification.

Each upstream track is evaluated with its own established fixtures. These
fixtures do not assert that their datasets, selections, or provenance coincide.
No protected-holdout targets are read and no selection is authorized here.
"""
from copy import deepcopy
import json

import pytest

from tests.test_m18_accuracy_profile_ranking import make_result, trusted_context
from tests.test_m19_d1_b2_b2_authorization_integration import (
    _internal as d1_internal,
    _c2_inputs as d1_c2_inputs,
    _decision as d1_decision,
)
from tests.test_m19_b2_i2_c3_evidence_integration import (
    integrated_args,
    temporal_for,
)
from zenerestimation.assessment.eligibility import VerificationStatus
from zenerestimation.assessment.evidence_integration import (
    TemporalSummaryStatus,
    integrate_assessment_evidence,
)
from zenerestimation.assessment.consolidated_assessment import build_consolidated_assessment
from zenerestimation.comparison.generalization_gap_assessment import assess_generalization_gap


# Register the transitive pytest fixture required by integrated_args.
from tests.test_m19_b2_i2_c1_alignment import case as c1_case


_DENIED = (
    "protected_holdout_assessment_permitted",
    "protected_holdout_comparison_permitted",
    "protected_holdout_ranking_permitted",
    "protected_holdout_disclosure_permitted",
)


def _d4():
    trusted = trusted_context()
    errors = {
        "LSTM": ((0.4, 0.4), (0.6, 0.6)),
        "GRU": ((0.1, 0.1), (0.2, 0.2)),
    }
    results = {name: make_result(name, trusted, errors=folds)
               for name, folds in errors.items()}
    return build_consolidated_assessment(results, validation_context=trusted)


def _d1(**overrides):
    args = {**d1_internal(), "benchmark_authorization_inputs": d1_c2_inputs()}
    args.update(overrides)
    return assess_generalization_gap(**args)


def _assert_d4_boundary(report):
    assert report["status"] == "available"
    assert report["internal_assessment"]["baseline_recommended_model"] == "GRU"
    assert report["internal_assessment"]["selection_authorized"] is False
    assert report["scientific_boundaries"] == {
        "selection_authorized": False,
        "prospective_certification_issued": False,
        "protected_holdout_accessed": False,
        "optimization_executed": False,
    }
    assert report["evidence_relationship"]["cross_track_identity_verified"] is False


def _assert_d1_boundary(report):
    assert report["status"] in {"AUTHORIZATION_REQUIRED", "AUTHORIZATION_REJECTED"}
    assert report["internal_evidence_verified"] is True
    assert report["comparability_verified"] is False
    assert report["holdout_assessed"] is False
    assert report["generalization_gap"] is None
    assert report["generalization_gap_metric"] is None
    for field in _DENIED:
        assert report[field] is False


def test_c1_01_available_internal_d4_and_rejected_real_d1_coexist():
    d4, d1 = _d4(), _d1()
    _assert_d4_boundary(d4)
    _assert_d1_boundary(d1)
    assert d1["status"] == "AUTHORIZATION_REJECTED"
    assert d1["benchmark_authorization_status"] == "AUTHORIZATION_REJECTED"


def test_c1_02_required_d1_still_denies_all_permissions(monkeypatch):
    # Controlled upstream-decision seam: this is NOT authenticated authorization.
    decision = d1_decision("AUTHORIZATION_REQUIRED")
    monkeypatch.setattr(
        "zenerestimation.comparison.generalization_authorization_integration."
        "assess_independent_benchmark_authorization",
        lambda **kwargs: deepcopy(decision),
    )
    d1 = _d1(supplied_authorization_decision=deepcopy(decision))
    _assert_d1_boundary(d1)
    assert d1["status"] == "AUTHORIZATION_REQUIRED"
    assert d1["benchmark_authorization_decision_digest"] == decision["decision_digest"]
    _assert_d4_boundary(_d4())


def test_c1_03_partial_observed_coverage_is_not_internal_ranking(integrated_args):
    observed = integrate_assessment_evidence(**integrated_args)
    snapshot = observed.to_dict()
    d4 = _d4()
    _assert_d4_boundary(d4)
    assert observed.coverage_status == "partial"
    assert observed.score_available is True
    assert snapshot["evidence"]["observability"]["coverage_mask"] == [True, False, False]
    assert snapshot["evidence"]["observed_metrics"]["eligible_count"] == 1
    assert d4["scientific_eligibility"] is None
    assert "observed_metrics" not in d4
    assert "baseline_recommended_model" not in snapshot


def test_c1_04_temporal_incompleteness_is_not_certification(integrated_args):
    args = dict(integrated_args)
    args["temporal_evidence"] = temporal_for(
        args["declaration"], {"chronology": VerificationStatus.UNKNOWN}
    )
    observed = integrate_assessment_evidence(**args)
    snapshot = observed.to_dict()
    assert observed.temporal_status is TemporalSummaryStatus.INCOMPLETE
    assert observed.coverage_status == "partial"
    assert observed.score_available is True
    assert snapshot["temporal_status"] == "incomplete"
    assert "scientific_classification" not in snapshot
    assert _d4()["scientific_boundaries"]["prospective_certification_issued"] is False


def test_c1_05_dataset_labels_do_not_authenticate_cross_track_identity(integrated_args):
    observed = integrate_assessment_evidence(**integrated_args)
    d4 = _d4()
    assert isinstance(observed.dataset_id, str)
    assert d4["evidence_relationship"]["cross_track_identity_verified"] is False
    assert "observed_evidence_authenticated" not in d4
    assert "cross_track_identity_verified" not in observed.to_dict()


def test_c1_06_d4_recommendation_is_not_a_d1_selection_artifact():
    d4 = _d4()
    d1_inputs = d1_internal()
    assert d4["internal_assessment"]["baseline_recommended_model"] == "GRU"
    assert "selection_artifact" not in d4
    assert "selection_artifact" in d1_inputs
    assert not isinstance(d1_inputs["selection_artifact"], str)
    _assert_d1_boundary(_d1(**d1_inputs))
    assert d4["scientific_boundaries"]["selection_authorized"] is False


def test_c1_07_contradictory_supplied_authorization_is_rejected():
    # The real C2 verifier rejects the intentionally invalid benchmark inputs.
    # A supplied 'required' decision cannot override that recomputation.
    d1 = _d1(supplied_authorization_decision=d1_decision("AUTHORIZATION_REQUIRED"))
    _assert_d1_boundary(d1)
    assert d1["status"] == "AUTHORIZATION_REJECTED"
    assert "SUPPLIED_AUTHORIZATION_DECISION_MISMATCH" in d1["reasons"]
    assert d1["benchmark_authorization_decision_digest"] is None


def test_c1_08_deterministic_detached_reports_and_no_input_mutation(integrated_args):
    original = {key: value.to_dict() for key, value in integrated_args.items()
                if hasattr(value, "to_dict")}
    first_d4, second_d4 = _d4(), _d4()
    first_d1, second_d1 = _d1(), _d1()
    first_obs = integrate_assessment_evidence(**integrated_args).to_dict()
    second_obs = integrate_assessment_evidence(**integrated_args).to_dict()
    for first, second in ((first_d4, second_d4),
                          (first_d1, second_d1),
                          (first_obs, second_obs)):
        assert first == second
        assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
            second, sort_keys=True, allow_nan=False
        )
    assert {key: value.to_dict() for key, value in integrated_args.items()
            if hasattr(value, "to_dict")} == original
    first_obs["evidence"]["observability"]["coverage_mask"][0] = False
    assert second_obs["evidence"]["observability"]["coverage_mask"][0] is True
