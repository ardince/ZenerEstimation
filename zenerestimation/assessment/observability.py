"""M19-B2-I2-B: immutable, declaration-bound target observability evidence.

Evidence findings are supplied by an external trusted verification process. This
module validates structure and consistency; it cannot authenticate measurements,
source references, availability timestamps, or prior target exposure.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from math import isfinite
from typing import Iterable

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, ExposureStatus, VerificationFinding, VerificationStatus,
    validate_assessment_declaration, validate_verification_finding,
)

OBSERVABILITY_CONTRACT_VERSION = "M19_B2_I2_B_OBSERVABILITY_V1"


class TargetOrigin(StrEnum):
    MEASURED = "measured"
    MISSING = "missing"
    INTERPOLATED = "interpolated"
    FORECAST_DERIVED = "forecast_derived"
    UNVERIFIED = "unverified"


class CoverageStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NONE = "none"


def _identifier(value: object, name: str) -> None:
    import re
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value) is None:
        raise ValueError(f"{name} must be a canonical identifier")


def _date(value: object, name: str) -> date:
    if type(value) is not str:
        raise ValueError(f"{name} must be canonical YYYY-MM-DD")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid date") from exc
    if result.isoformat() != value:
        raise ValueError(f"{name} must be canonical YYYY-MM-DD")
    return result


@dataclass(frozen=True, slots=True)
class TargetObservation:
    dataset_id: str
    target_date: str
    value_microVolt: float | None
    origin: TargetOrigin
    source_ref: str | None = None
    availability_date: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.dataset_id, "dataset_id")
        target = _date(self.target_date, "target_date")
        if target.day != 1 or target.month not in (3, 6, 9, 12):
            raise ValueError("target_date must be a quarterly March/June/September/December start")
        if type(self.origin) is not TargetOrigin:
            raise TypeError("origin must be TargetOrigin")
        if self.value_microVolt is not None:
            if type(self.value_microVolt) not in (int, float) or not isfinite(self.value_microVolt):
                raise ValueError("value_microVolt must be finite numeric or None")
        if self.source_ref is not None:
            _identifier(self.source_ref, "source_ref")
        if self.availability_date is not None:
            _date(self.availability_date, "availability_date")
        if self.origin is TargetOrigin.MISSING:
            if self.value_microVolt is not None or self.source_ref is not None or self.availability_date is not None:
                raise ValueError("missing target cannot carry a value, source or availability date")
        if self.origin is TargetOrigin.MEASURED:
            if self.value_microVolt is None or self.source_ref is None:
                raise ValueError("measured target requires a value and source_ref")
        if self.origin in (TargetOrigin.INTERPOLATED, TargetOrigin.FORECAST_DERIVED):
            if self.value_microVolt is None:
                raise ValueError("derived target requires a finite value")

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "target_date": self.target_date,
            "value_microVolt": self.value_microVolt,
            "origin": self.origin.value,
            "source_ref": self.source_ref,
            "availability_date": self.availability_date,
        }


@dataclass(frozen=True, slots=True)
class TargetObservationEvidence:
    observation: TargetObservation
    provenance: VerificationFinding
    availability: VerificationFinding
    exposure: VerificationFinding
    exposure_status: ExposureStatus = ExposureStatus.UNKNOWN

    def __post_init__(self) -> None:
        if not isinstance(self.observation, TargetObservation):
            raise TypeError("observation must be TargetObservation")
        for name in ("provenance", "availability", "exposure"):
            validate_verification_finding(getattr(self, name))
        if type(self.exposure_status) is not ExposureStatus:
            raise TypeError("exposure_status must be ExposureStatus")
        if self.exposure_status is ExposureStatus.UNKNOWN and self.exposure.status is VerificationStatus.VERIFIED:
            raise ValueError("verified exposure finding requires known exposure status")
        if self.exposure_status is not ExposureStatus.UNKNOWN and self.exposure.status is not VerificationStatus.VERIFIED:
            raise ValueError("known exposure status requires verified exposure finding")
        if self.observation.origin is TargetOrigin.MISSING:
            if self.provenance.status is VerificationStatus.VERIFIED:
                raise ValueError("missing target cannot have verified measurement provenance")
            if self.availability.status is VerificationStatus.VERIFIED:
                raise ValueError("missing target cannot have verified measurement availability")
        if self.observation.origin is not TargetOrigin.MEASURED and self.provenance.status is VerificationStatus.VERIFIED:
            raise ValueError("non-measured target cannot have verified measurement provenance")
        if self.availability.status is VerificationStatus.VERIFIED and self.observation.availability_date is None:
            raise ValueError("verified availability requires availability_date")

    @property
    def verified_observed(self) -> bool:
        """Conditional on caller-supplied provenance finding; not authentication."""
        return (self.observation.origin is TargetOrigin.MEASURED
                and self.provenance.status is VerificationStatus.VERIFIED)

    def to_dict(self) -> dict[str, object]:
        return {
            "observation": self.observation.to_dict(),
            "provenance": self.provenance.to_dict(),
            "availability": self.availability.to_dict(),
            "exposure": self.exposure.to_dict(),
            "exposure_status": self.exposure_status.value,
            "verified_observed": self.verified_observed,
        }


@dataclass(frozen=True, slots=True)
class TargetObservabilityReport:
    assessment_id: str
    dataset_id: str
    declaration_digest: str
    declared_horizon: int
    observations: tuple[TargetObservationEvidence, ...]
    contract_version: str = OBSERVABILITY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _identifier(self.assessment_id, "assessment_id")
        _identifier(self.dataset_id, "dataset_id")
        import re
        if type(self.declaration_digest) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", self.declaration_digest) is None:
            raise ValueError("declaration_digest must be canonical sha256")
        if type(self.declared_horizon) is not int or self.declared_horizon <= 0:
            raise ValueError("declared_horizon must be positive integer")
        if self.contract_version != OBSERVABILITY_CONTRACT_VERSION:
            raise ValueError("unsupported observability contract version")
        if type(self.observations) is not tuple or len(self.observations) != self.declared_horizon:
            raise ValueError("observations must be a complete tuple of declared horizon length")
        dates = []
        for item in self.observations:
            if not isinstance(item, TargetObservationEvidence):
                raise TypeError("observations must contain TargetObservationEvidence")
            if item.observation.dataset_id != self.dataset_id:
                raise ValueError("observation dataset mismatch")
            dates.append(item.observation.target_date)
        if dates != sorted(set(dates)):
            raise ValueError("observations must have unique dates in canonical order")

    @property
    def coverage_mask(self) -> tuple[bool, ...]:
        return tuple(item.verified_observed for item in self.observations)

    @property
    def observed_count(self) -> int:
        return sum(self.coverage_mask)

    @property
    def missing_count(self) -> int:
        """Number of quarters without verified observed measurement (not just MISSING origin)."""
        return self.declared_horizon - self.observed_count

    @property
    def coverage_status(self) -> CoverageStatus:
        if self.observed_count == 0:
            return CoverageStatus.NONE
        if self.observed_count == self.declared_horizon:
            return CoverageStatus.COMPLETE
        return CoverageStatus.PARTIAL

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "assessment_id": self.assessment_id,
            "dataset_id": self.dataset_id,
            "declaration_digest": self.declaration_digest,
            "declared_horizon": self.declared_horizon,
            "observations": [item.to_dict() for item in self.observations],
            "coverage_mask": list(self.coverage_mask),
            "observed_count": self.observed_count,
            "missing_count": self.missing_count,
            "coverage_status": self.coverage_status.value,
        }


def build_target_observability_report(
    declaration: AssessmentDeclaration,
    observations: Iterable[TargetObservationEvidence],
) -> TargetObservabilityReport:
    """Validate exact calendar coverage and return an immutable evidence report.

    Never infer provenance or independent exposure from numeric measurements.
    Caller must supply findings from an appropriate verification process.
    """
    validate_assessment_declaration(declaration)
    entries = tuple(observations)
    if len(entries) != len(declaration.forecast_dates):
        raise ValueError("every declared forecast quarter requires one observation record")
    if any(not isinstance(item, TargetObservationEvidence) for item in entries):
        raise TypeError("observations must contain TargetObservationEvidence")
    dates = [item.observation.target_date for item in entries]
    if dates != list(declaration.forecast_dates):
        raise ValueError("observation dates must match declaration exactly and in order")
    return TargetObservabilityReport(
        assessment_id=declaration.assessment_id,
        dataset_id=declaration.dataset_id,
        declaration_digest=declaration.digest,
        declared_horizon=len(declaration.forecast_dates),
        observations=entries,
    )
