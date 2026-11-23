"""M19-B2-I2-B immutable observation boundary and negative tests."""
from dataclasses import FrozenInstanceError

import pytest

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION, ExposureStatus,
    VerificationFinding, VerificationStatus,
)
from zenerestimation.assessment.observability import (
    CoverageStatus, TargetObservation, TargetObservationEvidence, TargetOrigin,
    TargetObservabilityReport, build_target_observability_report,
)

DATES = ("2025-03-01", "2025-06-01", "2025-09-01", "2025-12-01", "2026-03-01", "2026-06-01")


def declaration():
    return AssessmentDeclaration(CONTRACT_VERSION, "assess.410", AssessmentTrack.RETROSPECTIVE,
                                 "410", "2024-12-01", DATES)


def finding(status=VerificationStatus.UNKNOWN, reason="UNKNOWN"):
    if status is VerificationStatus.VERIFIED:
        return VerificationFinding(status, "evidence.1", "verifier.1")
    if status is VerificationStatus.FAILED:
        return VerificationFinding(status, reason_codes=(reason,))
    return VerificationFinding(status)


def entry(date=DATES[0], origin=TargetOrigin.MEASURED, value=31.0,
          source="measurement.1", provenance=VerificationStatus.VERIFIED,
          availability=VerificationStatus.UNKNOWN, availability_date=None,
          exposure_status=ExposureStatus.UNKNOWN, exposure=VerificationStatus.UNKNOWN,
          dataset="410"):
    return TargetObservationEvidence(
        TargetObservation(dataset, date, value, origin, source, availability_date),
        finding(provenance), finding(availability), finding(exposure), exposure_status,
    )


def missing(date):
    return entry(date, TargetOrigin.MISSING, None, None, VerificationStatus.UNKNOWN)


def full():
    return tuple(entry(date=d) for d in DATES)


def test_complete_verified_measurements():
    report = build_target_observability_report(declaration(), full())
    assert report.observed_count == 6
    assert report.missing_count == 0
    assert report.coverage_status is CoverageStatus.COMPLETE
    assert report.coverage_mask == (True,) * 6
    assert report.declaration_digest == declaration().digest


def test_partial_coverage_preserves_missing_quarters():
    rows = (entry(date=DATES[0]), entry(date=DATES[1]), missing(DATES[2]),
            entry(date=DATES[3]), entry(date=DATES[4]), missing(DATES[5]))
    report = build_target_observability_report(declaration(), rows)
    assert report.coverage_mask == (True, True, False, True, True, False)
    assert (report.observed_count, report.missing_count) == (4, 2)
    assert report.coverage_status is CoverageStatus.PARTIAL


def test_zero_coverage():
    report = build_target_observability_report(declaration(), tuple(missing(d) for d in DATES))
    assert report.coverage_status is CoverageStatus.NONE


def test_dataset_identity_mismatch():
    with pytest.raises(ValueError, match="dataset mismatch"):
        build_target_observability_report(declaration(), (entry(dataset="110"), *full()[1:]))


def test_outside_forecast_calendar_rejected():
    with pytest.raises(ValueError, match="dates must match"):
        build_target_observability_report(declaration(), (*full()[:-1], entry(date="2026-09-01")))


def test_duplicate_date_rejected():
    with pytest.raises(ValueError, match="dates must match"):
        build_target_observability_report(declaration(), (*full()[:-1], entry(date=DATES[-2])))


def test_missing_with_finite_value_rejected():
    with pytest.raises(ValueError, match="missing target"):
        entry(origin=TargetOrigin.MISSING, value=2.0, source=None, provenance=VerificationStatus.UNKNOWN)


def test_measured_nan_rejected():
    with pytest.raises(ValueError, match="finite"):
        entry(value=float("nan"))


def test_interpolation_cannot_claim_verified_measurement_provenance():
    with pytest.raises(ValueError, match="non-measured"):
        entry(origin=TargetOrigin.INTERPOLATED)


def test_interpolation_excluded_even_when_finite():
    row = entry(origin=TargetOrigin.INTERPOLATED, provenance=VerificationStatus.UNKNOWN)
    report = build_target_observability_report(declaration(), (row, *full()[1:]))
    assert report.coverage_mask[0] is False


def test_forecast_derived_excluded():
    row = entry(origin=TargetOrigin.FORECAST_DERIVED, provenance=VerificationStatus.UNKNOWN)
    assert not row.verified_observed


def test_unknown_provenance_not_verified_observed():
    row = entry(provenance=VerificationStatus.UNKNOWN)
    assert not row.verified_observed


def test_source_reference_alone_not_proof():
    row = entry(source="source.claim", provenance=VerificationStatus.UNKNOWN)
    assert row.observation.source_ref == "source.claim"
    assert not row.verified_observed


def test_availability_unknown_independent_from_provenance():
    row = entry()
    assert row.verified_observed
    assert row.availability.status is VerificationStatus.UNKNOWN


def test_verified_availability_requires_date():
    with pytest.raises(ValueError, match="availability_date"):
        entry(availability=VerificationStatus.VERIFIED)


def test_known_exposure_preserved_without_disqualifying_measurement():
    row = entry(exposure_status=ExposureStatus.EXPOSED, exposure=VerificationStatus.VERIFIED)
    assert row.verified_observed
    assert row.exposure_status is ExposureStatus.EXPOSED


def test_known_exposure_requires_verified_finding():
    with pytest.raises(ValueError, match="known exposure"):
        entry(exposure_status=ExposureStatus.NOT_EXPOSED)


def test_verified_exposure_requires_known_status():
    with pytest.raises(ValueError, match="verified exposure"):
        entry(exposure=VerificationStatus.VERIFIED)


def test_omitted_quarter_rejected():
    with pytest.raises(ValueError, match="every declared"):
        build_target_observability_report(declaration(), full()[:-1])


def test_reordered_evidence_rejected():
    rows = list(full())
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="dates must match"):
        build_target_observability_report(declaration(), rows)


def test_deterministic_serialization():
    assert build_target_observability_report(declaration(), full()).to_dict() == build_target_observability_report(declaration(), full()).to_dict()


def test_immutable_observation():
    with pytest.raises(FrozenInstanceError):
        entry().observation.value_microVolt = 5.0


def test_immutable_report_and_observation_collection():
    report = build_target_observability_report(declaration(), full())
    assert type(report.observations) is tuple
    with pytest.raises(FrozenInstanceError):
        report.assessment_id = "changed"


def test_noncanonical_quarter_rejected():
    with pytest.raises(ValueError, match="quarterly"):
        entry(date="2025-04-01")


def test_nonfinite_infinity_rejected():
    with pytest.raises(ValueError, match="finite"):
        entry(value=float("inf"))


def test_unverified_origin_not_observed():
    row = entry(origin=TargetOrigin.UNVERIFIED, provenance=VerificationStatus.UNKNOWN)
    assert not row.verified_observed


def test_missing_cannot_claim_verified_availability():
    with pytest.raises(ValueError, match="missing target"):
        entry(origin=TargetOrigin.MISSING, value=None, source=None,
              provenance=VerificationStatus.UNKNOWN, availability=VerificationStatus.VERIFIED,
              availability_date="2025-04-01")


def test_malformed_origin_rejected():
    with pytest.raises(TypeError, match="TargetOrigin"):
        entry(origin="measured")


def test_no_accuracy_metrics_in_report():
    result = build_target_observability_report(declaration(), full()).to_dict()
    assert "rmse" not in result and "mae" not in result


def test_report_constructor_rejects_noncanonical_order():
    with pytest.raises(ValueError, match="canonical order"):
        TargetObservabilityReport("assess.410", "410", declaration().digest, 2,
                                  (entry(date=DATES[1]), entry(date=DATES[0])))
