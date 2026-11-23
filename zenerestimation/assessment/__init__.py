"""M19 scientific assessment contracts (structural validation only)."""
from .eligibility import (
    CONTRACT_VERSION, AssessmentDeclaration, AssessmentTrack, AssessmentValidity,
    ExposureStatus, ScientificClassification, TrustedVerificationEvidence,
    VerificationFinding, VerificationStatus, validate_assessment_declaration,
    validate_evidence_binding, validate_verification_evidence,
    validate_verification_finding,
)

__all__ = [
    "CONTRACT_VERSION", "AssessmentDeclaration", "AssessmentTrack",
    "AssessmentValidity", "ExposureStatus", "ScientificClassification",
    "TrustedVerificationEvidence", "VerificationFinding", "VerificationStatus",
    "validate_assessment_declaration", "validate_evidence_binding",
    "validate_verification_evidence", "validate_verification_finding",
]

from .decision import ScientificEligibilityDecision, evaluate_scientific_eligibility
__all__ += ["ScientificEligibilityDecision", "evaluate_scientific_eligibility"]
