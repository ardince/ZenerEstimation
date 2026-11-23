"""M19-B2-I1-B pure eligibility policy; never authenticates or certifies evidence."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .eligibility import (
    CONTRACT_VERSION, AssessmentDeclaration, AssessmentTrack, AssessmentValidity,
    ExposureStatus, ScientificClassification, TrustedVerificationEvidence,
    VerificationStatus, validate_evidence_binding,
)

_CHECKS = (
    ("technical_integrity", "TECHNICAL_INTEGRITY"),
    ("protocol_compliance", "PROTOCOL_COMPLIANCE"),
    ("frozen_artifact_integrity", "FROZEN_ARTIFACT_INTEGRITY"),
)
_PROSPECTIVE = (
    ("forecast_freeze", "FORECAST_FREEZE"),
    ("target_availability", "TARGET_AVAILABILITY"),
    ("chronology", "FREEZE_CHRONOLOGY"),
)


@dataclass(frozen=True, slots=True)
class ScientificEligibilityDecision:
    """Policy conclusion conditional on supplied evidence, NOT an authenticated certificate."""

    contract_version: str
    assessment_id: str
    assessment_validity: AssessmentValidity
    scientific_classification: ScientificClassification | None
    prospective_independence_eligible: bool
    prospective_independence_certified: bool
    reason_codes: tuple[str, ...]
    evidence_refs: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.prospective_independence_certified:
            raise ValueError("pure policy decisions cannot issue certification")
        if self.assessment_validity is not AssessmentValidity.VALID and self.scientific_classification is not None:
            raise ValueError("invalid or incomplete assessments cannot be classified")
        if self.prospective_independence_eligible != (
            self.assessment_validity is AssessmentValidity.VALID
            and self.scientific_classification is ScientificClassification.PROSPECTIVE_INDEPENDENT
        ):
            raise ValueError("prospective eligibility must match valid prospective classification")
        object.__setattr__(self, "reason_codes", tuple(sorted(set(self.reason_codes))))
        object.__setattr__(self, "evidence_refs", MappingProxyType(dict(sorted(self.evidence_refs.items()))))

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "assessment_id": self.assessment_id,
            "assessment_validity": self.assessment_validity.value,
            "scientific_classification": None if self.scientific_classification is None else self.scientific_classification.value,
            "prospective_independence_eligible": self.prospective_independence_eligible,
            "prospective_independence_certified": False,
            "reason_codes": list(self.reason_codes),
            "evidence_refs": dict(self.evidence_refs),
        }


def evaluate_scientific_eligibility(
    declaration: AssessmentDeclaration,
    evidence: TrustedVerificationEvidence,
) -> ScientificEligibilityDecision:
    """Deterministic, fail-closed classification of *supplied* verification findings.

    Raises on malformed or mismatched contracts. Never verifies authenticity of
    evidence refs, timestamps, digests or claimed verifier identities.
    """
    validate_evidence_binding(declaration, evidence)
    reasons: set[str] = set()
    refs = {name: finding.evidence_ref for name, finding in (
        ("technical_integrity", evidence.technical_integrity),
        ("protocol_compliance", evidence.protocol_compliance),
        ("exposure_evidence", evidence.exposure_evidence),
        ("forecast_freeze", evidence.forecast_freeze),
        ("target_availability", evidence.target_availability),
        ("chronology", evidence.chronology),
        ("frozen_artifact_integrity", evidence.frozen_artifact_integrity),
    ) if finding.evidence_ref is not None}

    def decision(validity: AssessmentValidity, classification: ScientificClassification | None):
        return ScientificEligibilityDecision(
            contract_version=CONTRACT_VERSION,
            assessment_id=declaration.assessment_id,
            assessment_validity=validity,
            scientific_classification=classification,
            prospective_independence_eligible=(classification is ScientificClassification.PROSPECTIVE_INDEPENDENT),
            prospective_independence_certified=False,
            reason_codes=tuple(sorted(reasons)),
            evidence_refs=refs,
        )

    for attr, label in _CHECKS:
        finding = getattr(evidence, attr)
        if finding.status is VerificationStatus.FAILED:
            reasons.add(f"{label}_FAILED")
        elif finding.status is VerificationStatus.UNKNOWN:
            reasons.add(f"{label}_UNKNOWN")
    if evidence.exposure_evidence.status is VerificationStatus.FAILED:
        reasons.add("EXPOSURE_VERIFICATION_FAILED")
    if any(x.endswith("_FAILED") for x in reasons):
        return decision(AssessmentValidity.INVALID, None)
    if any(x.endswith("_UNKNOWN") for x in reasons):
        return decision(AssessmentValidity.INCOMPLETE, None)

    if evidence.target_exposure is ExposureStatus.EXPOSED:
        reasons.add("TARGET_EXPOSURE_CONFIRMED")
    elif evidence.target_exposure is ExposureStatus.UNKNOWN:
        reasons.add("TARGET_EXPOSURE_UNKNOWN")

    if declaration.assessment_track is AssessmentTrack.RETROSPECTIVE:
        # An unknown exposure history must not be upgraded to a controlled
        # retrospective classification merely because the track says retrospective.
        if evidence.target_exposure is ExposureStatus.UNKNOWN:
            return decision(AssessmentValidity.VALID, ScientificClassification.INDEPENDENCE_UNVERIFIED)
        return decision(AssessmentValidity.VALID, ScientificClassification.CONTROLLED_RETROSPECTIVE)

    if evidence.target_exposure is ExposureStatus.EXPOSED:
        reasons.add("ASSESSMENT_TRACK_MISMATCH")
        return decision(AssessmentValidity.INVALID, None)

    for attr, label in _PROSPECTIVE:
        finding = getattr(evidence, attr)
        if finding.status is VerificationStatus.FAILED:
            reasons.add(f"{label}_FAILED")
        elif finding.status is VerificationStatus.UNKNOWN:
            reasons.add(f"{label}_UNVERIFIED")
    if any(x.endswith("_FAILED") for x in reasons):
        return decision(AssessmentValidity.INVALID, None)
    if reasons:
        return decision(AssessmentValidity.VALID, ScientificClassification.INDEPENDENCE_UNVERIFIED)
    return decision(AssessmentValidity.VALID, ScientificClassification.PROSPECTIVE_INDEPENDENT)
