"""M19-B2-I1-D narrow hardening regression tests; synthetic records only."""
from dataclasses import replace

import pytest

from zenerestimation.assessment import (
    AssessmentDeclaration, AssessmentTrack, AssessmentValidity,
    ExposureStatus, ScientificClassification, VerificationStatus,
    evaluate_scientific_eligibility,
)
from tests.test_m19_b2_i1_c_negative import finding, pair


@pytest.mark.parametrize("dates", [
    ("2026-01-01", "2026-04-01"),
    ("2026-03-31", "2026-06-30"),
    ("2026-03-02", "2026-06-02"),
    ("2026-01-31", "2026-04-30"),
    ("2026-03-01", "2026-06-02"),
])
def test_d01_forecast_dates_must_use_march_quarter_starts(dates):
    d, _ = pair()
    with pytest.raises(ValueError, match="first-of-month"):
        replace(d, forecast_dates=dates)


def test_d02_standard_quarter_starts_still_accepted():
    d, e = pair()
    assert d.forecast_dates == ("2025-06-01", "2025-09-01")
    assert evaluate_scientific_eligibility(d, e).prospective_independence_eligible


@pytest.mark.parametrize("track", [AssessmentTrack.RETROSPECTIVE, AssessmentTrack.PROSPECTIVE])
def test_d03_failed_exposure_verification_invalidates(track):
    d, e = pair(track=track, exposure=ExposureStatus.UNKNOWN,
                exposure_evidence=finding(VerificationStatus.FAILED, "exposure"))
    r = evaluate_scientific_eligibility(d, e)
    assert r.assessment_validity is AssessmentValidity.INVALID
    assert r.scientific_classification is None
    assert "EXPOSURE_VERIFICATION_FAILED" in r.reason_codes
    assert not r.prospective_independence_eligible
    assert not r.prospective_independence_certified


def test_d04_unknown_exposure_verification_remains_unverified():
    d, e = pair(exposure=ExposureStatus.UNKNOWN)
    r = evaluate_scientific_eligibility(d, e)
    assert r.assessment_validity is AssessmentValidity.VALID
    assert r.scientific_classification is ScientificClassification.INDEPENDENCE_UNVERIFIED
    assert "EXPOSURE_VERIFICATION_FAILED" not in r.reason_codes


def test_d05_failed_exposure_precedes_unknown_core():
    d, e = pair(exposure=ExposureStatus.UNKNOWN,
                exposure_evidence=finding(VerificationStatus.FAILED, "exposure"),
                technical_integrity=finding(VerificationStatus.UNKNOWN, "tech"))
    r = evaluate_scientific_eligibility(d, e)
    assert r.assessment_validity is AssessmentValidity.INVALID
    assert "EXPOSURE_VERIFICATION_FAILED" in r.reason_codes
    assert "TECHNICAL_INTEGRITY_UNKNOWN" in r.reason_codes
