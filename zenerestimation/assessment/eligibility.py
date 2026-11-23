"""M19-B2-I1-A: structural scientific-assessment evidence contracts.

These types do not authenticate evidence or issue scientific certification.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

CONTRACT_VERSION = "M19_B2_SCIENTIFIC_ELIGIBILITY_V1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_REASON = re.compile(r"^[A-Z][A-Z0-9_]*$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class AssessmentTrack(StrEnum):
    RETROSPECTIVE = "retrospective"
    PROSPECTIVE = "prospective"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ExposureStatus(StrEnum):
    EXPOSED = "exposed"
    NOT_EXPOSED = "not_exposed"
    UNKNOWN = "unknown"


class AssessmentValidity(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    INCOMPLETE = "incomplete"


class ScientificClassification(StrEnum):
    CONTROLLED_RETROSPECTIVE = "controlled_retrospective"
    INDEPENDENCE_UNVERIFIED = "independence_unverified"
    PROSPECTIVE_INDEPENDENT = "prospective_independent"


def _enum(value: object, kind: type[StrEnum], name: str) -> None:
    if type(value) is not kind:
        raise TypeError(f"{name} must be {kind.__name__}")


def _identifier(value: object, name: str) -> None:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a nonempty canonical identifier")


def _calendar_date(value: object, name: str) -> date:
    if type(value) is not str or not _DATE.fullmatch(value):
        raise ValueError(f"{name} must be YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid calendar date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must be canonical YYYY-MM-DD")
    return parsed


def _next_quarter(value: date) -> date:
    month = value.month + 3
    year = value.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return date(year, month, value.day)


def _refs(value: object, name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    output: dict[str, str] = {}
    for key, ref in value.items():
        _identifier(key, f"{name} key")
        _identifier(ref, f"{name}[{key}]")
        output[key] = ref
    return MappingProxyType(dict(sorted(output.items())))


def _reasons(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError("reason_codes must be a tuple")
    for code in value:
        if type(code) is not str or not _REASON.fullmatch(code):
            raise ValueError("reason_codes must contain canonical uppercase codes")
    if len(value) != len(set(value)):
        raise ValueError("reason_codes must be unique")
    return tuple(sorted(value))


@dataclass(frozen=True, slots=True)
class AssessmentDeclaration:
    contract_version: str
    assessment_id: str
    assessment_track: AssessmentTrack
    dataset_id: str
    development_cutoff: str
    forecast_dates: tuple[str, ...]
    evidence_refs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_assessment_declaration(self)
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, "evidence_refs"))

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "assessment_id": self.assessment_id,
            "assessment_track": self.assessment_track.value,
            "dataset_id": self.dataset_id,
            "development_cutoff": self.development_cutoff,
            "forecast_dates": list(self.forecast_dates),
            "evidence_refs": dict(self.evidence_refs),
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class VerificationFinding:
    status: VerificationStatus
    evidence_ref: str | None = None
    verifier_id: str | None = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_verification_finding(self)
        object.__setattr__(self, "reason_codes", _reasons(self.reason_codes))

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status.value, "evidence_ref": self.evidence_ref,
                "verifier_id": self.verifier_id, "reason_codes": list(self.reason_codes)}


@dataclass(frozen=True, slots=True)
class TrustedVerificationEvidence:
    assessment_id: str
    dataset_id: str
    declaration_digest: str
    technical_integrity: VerificationFinding
    protocol_compliance: VerificationFinding
    target_exposure: ExposureStatus
    exposure_evidence: VerificationFinding
    forecast_freeze: VerificationFinding
    target_availability: VerificationFinding
    chronology: VerificationFinding
    frozen_artifact_integrity: VerificationFinding

    def __post_init__(self) -> None:
        validate_verification_evidence(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "assessment_id": self.assessment_id,
            "dataset_id": self.dataset_id,
            "declaration_digest": self.declaration_digest,
            "technical_integrity": self.technical_integrity.to_dict(),
            "protocol_compliance": self.protocol_compliance.to_dict(),
            "target_exposure": self.target_exposure.value,
            "exposure_evidence": self.exposure_evidence.to_dict(),
            "forecast_freeze": self.forecast_freeze.to_dict(),
            "target_availability": self.target_availability.to_dict(),
            "chronology": self.chronology.to_dict(),
            "frozen_artifact_integrity": self.frozen_artifact_integrity.to_dict(),
        }


def validate_assessment_declaration(declaration: AssessmentDeclaration) -> None:
    if not isinstance(declaration, AssessmentDeclaration):
        raise TypeError("declaration must be AssessmentDeclaration")
    if declaration.contract_version != CONTRACT_VERSION:
        raise ValueError("unsupported scientific eligibility contract version")
    _identifier(declaration.assessment_id, "assessment_id")
    _identifier(declaration.dataset_id, "dataset_id")
    _enum(declaration.assessment_track, AssessmentTrack, "assessment_track")
    cutoff = _calendar_date(declaration.development_cutoff, "development_cutoff")
    if type(declaration.forecast_dates) is not tuple or not declaration.forecast_dates:
        raise ValueError("forecast_dates must be a nonempty tuple")
    dates = [_calendar_date(v, "forecast_dates entry") for v in declaration.forecast_dates]
    if dates[0] <= cutoff:
        raise ValueError("forecast dates must be after development cutoff")
    if any(d.day != 1 or d.month not in (3, 6, 9, 12) for d in dates):
        raise ValueError("forecast dates must be first-of-month March/June/September/December three-month quarters")
    if any(b <= a for a, b in zip(dates, dates[1:])):
        raise ValueError("forecast dates must be strictly increasing and unique")
    if any(b != _next_quarter(a) for a, b in zip(dates, dates[1:])):
        raise ValueError("forecast dates must be consecutive three-month periods")
    _refs(declaration.evidence_refs, "evidence_refs")


def validate_verification_finding(finding: VerificationFinding) -> None:
    if not isinstance(finding, VerificationFinding):
        raise TypeError("finding must be VerificationFinding")
    _enum(finding.status, VerificationStatus, "status")
    if finding.evidence_ref is not None:
        _identifier(finding.evidence_ref, "evidence_ref")
    if finding.verifier_id is not None:
        _identifier(finding.verifier_id, "verifier_id")
    reasons = _reasons(finding.reason_codes)
    if finding.status is VerificationStatus.VERIFIED:
        if finding.evidence_ref is None or finding.verifier_id is None:
            raise ValueError("verified finding requires evidence_ref and verifier_id")
    if finding.status is VerificationStatus.FAILED and not reasons:
        raise ValueError("failed finding requires at least one reason code")


def validate_verification_evidence(evidence: TrustedVerificationEvidence) -> None:
    if not isinstance(evidence, TrustedVerificationEvidence):
        raise TypeError("evidence must be TrustedVerificationEvidence")
    _identifier(evidence.assessment_id, "assessment_id")
    _identifier(evidence.dataset_id, "dataset_id")
    if (type(evidence.declaration_digest) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", evidence.declaration_digest) is None):
        raise ValueError("declaration_digest must be canonical sha256 digest")
    _enum(evidence.target_exposure, ExposureStatus, "target_exposure")
    for name in ("technical_integrity", "protocol_compliance", "exposure_evidence",
                 "forecast_freeze", "target_availability", "chronology",
                 "frozen_artifact_integrity"):
        validate_verification_finding(getattr(evidence, name))
    if (evidence.target_exposure is not ExposureStatus.UNKNOWN
            and evidence.exposure_evidence.status is not VerificationStatus.VERIFIED):
        # Keep the distinction explicit; an unverified non-exposure claim cannot
        # masquerade as a verified exposure conclusion.
        raise ValueError("known exposure status requires verified exposure evidence")
    if (evidence.target_exposure is ExposureStatus.UNKNOWN
            and evidence.exposure_evidence.status is VerificationStatus.VERIFIED):
        raise ValueError("verified exposure finding must establish exposure status")


def validate_evidence_binding(
    declaration: AssessmentDeclaration, evidence: TrustedVerificationEvidence,
) -> None:
    """Check structural binding, NOT authenticity of the supplied evidence."""
    validate_assessment_declaration(declaration)
    validate_verification_evidence(evidence)
    if declaration.assessment_id != evidence.assessment_id:
        raise ValueError("assessment_id mismatch")
    if declaration.dataset_id != evidence.dataset_id:
        raise ValueError("dataset_id mismatch")
    if declaration.digest != evidence.declaration_digest:
        raise ValueError("declaration digest mismatch")
