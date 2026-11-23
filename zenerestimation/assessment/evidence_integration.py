"""M19-B2-I2-C3: read-only integration of conditional assessment evidence.

This module checks cross-contract consistency. It neither authenticates externally
supplied findings nor establishes scientific independence or certification.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, VerificationStatus, validate_assessment_declaration,
)
from zenerestimation.assessment.temporal import (
    TEMPORAL_CONTRACT_VERSION, TemporalEligibilityEvidence,
)
from zenerestimation.assessment.observability import (
    OBSERVABILITY_CONTRACT_VERSION, TargetObservabilityReport,
    build_target_observability_report,
)
from zenerestimation.assessment.observed_scoring import (
    ALIGNMENT_CONTRACT_VERSION, ObservedAlignmentResult,
)
from zenerestimation.assessment.observed_metrics import (
    SCORING_CONTRACT_VERSION, ObservedMetricResult, score_verified_observations,
)

INTEGRATION_CONTRACT_VERSION = "M19_B2_I2_C3_EVIDENCE_INTEGRATION_V1"
_TEMPORAL_FIELDS = (
    "plan_binding", "execution_integrity", "training_cutoff",
    "development_boundary", "forecast_freeze", "target_availability", "chronology",
)


class TemporalSummaryStatus(StrEnum):
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    VERIFIED_FINDINGS = "verified_findings"


def _temporal_status(evidence: TemporalEligibilityEvidence) -> TemporalSummaryStatus:
    statuses = tuple(getattr(evidence, name).status for name in _TEMPORAL_FIELDS)
    if VerificationStatus.FAILED in statuses:
        return TemporalSummaryStatus.FAILED
    if VerificationStatus.UNKNOWN in statuses:
        return TemporalSummaryStatus.INCOMPLETE
    return TemporalSummaryStatus.VERIFIED_FINDINGS


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True, slots=True)
class IntegratedAssessmentEvidence:
    """Immutable serialized evidence snapshot, with typed source access.

    The typed component objects are retained for inspection, but C2 contains
    nested mutable dictionaries. Consequently `to_dict()` is always generated
    from the sealed JSON snapshot, not from potentially mutated source objects.
    Consumers requiring a tamper-resistant boundary should use `to_dict()` and
    verify the provenance of the supplied upstream evidence separately.
    """

    contract_version: str
    assessment_id: str
    dataset_id: str
    declaration_digest: str
    execution_artifact_digest: str
    temporal_status: TemporalSummaryStatus
    coverage_status: str
    score_available: bool
    temporal_evidence: TemporalEligibilityEvidence = field(repr=False)
    observability_report: TargetObservabilityReport = field(repr=False)
    alignment_result: ObservedAlignmentResult = field(repr=False)
    metric_result: ObservedMetricResult = field(repr=False)
    _snapshot_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.contract_version != INTEGRATION_CONTRACT_VERSION:
            raise ValueError("unsupported C3 integration contract")
        payload = {
            "contract_version": self.contract_version,
            "assessment_id": self.assessment_id,
            "dataset_id": self.dataset_id,
            "declaration_digest": self.declaration_digest,
            "execution_artifact_digest": self.execution_artifact_digest,
            "temporal_status": self.temporal_status.value,
            "coverage_status": self.coverage_status,
            "score_available": self.score_available,
            "evidence": {
                "temporal": self.temporal_evidence.to_dict(),
                "observability": self.observability_report.to_dict(),
                "alignment": self.alignment_result.to_dict(),
                "observed_metrics": self.metric_result.to_dict(),
            },
        }
        object.__setattr__(self, "_snapshot_json", _canonical_json(payload))

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh, deterministic, detached JSON-compatible snapshot."""
        return json.loads(self._snapshot_json)


def integrate_assessment_evidence(
    *,
    declaration: AssessmentDeclaration,
    temporal_evidence: TemporalEligibilityEvidence,
    observability_report: TargetObservabilityReport,
    alignment_result: ObservedAlignmentResult,
    metric_result: ObservedMetricResult,
) -> IntegratedAssessmentEvidence:
    """Integrate I2-A/B/C1/C2 without promoting evidence or recalculating forecasts.

    Reject inconsistent source objects, including forged C2 metrics and rows.
    Temporal failure is retained as a finding rather than suppressing the
    separately meaningful observed-only accuracy result.
    """
    validate_assessment_declaration(declaration)
    expected_types = (
        (temporal_evidence, TemporalEligibilityEvidence, "temporal_evidence"),
        (observability_report, TargetObservabilityReport, "observability_report"),
        (alignment_result, ObservedAlignmentResult, "alignment_result"),
        (metric_result, ObservedMetricResult, "metric_result"),
    )
    for value, cls, name in expected_types:
        if not isinstance(value, cls):
            raise TypeError(f"{name} must be {cls.__name__}")

    for value, version, name in (
        (temporal_evidence, TEMPORAL_CONTRACT_VERSION, "temporal"),
        (observability_report, OBSERVABILITY_CONTRACT_VERSION, "observability"),
        (alignment_result, ALIGNMENT_CONTRACT_VERSION, "alignment"),
        (metric_result, SCORING_CONTRACT_VERSION, "metrics"),
    ):
        if value.contract_version != version:
            raise ValueError(f"unsupported {name} contract version")
        if value.assessment_id != declaration.assessment_id:
            raise ValueError(f"{name} assessment_id mismatch")
        if value.declaration_digest != declaration.digest:
            raise ValueError(f"{name} declaration_digest mismatch")

    for value, name in (
        (observability_report, "observability"),
        (alignment_result, "alignment"),
        (metric_result, "metrics"),
    ):
        if value.dataset_id != declaration.dataset_id:
            raise ValueError(f"{name} dataset_id mismatch")

    # Rebuild B against the exact declaration, rather than trusting only its
    # constructor (which does not itself bind the dates to the declaration).
    expected_report = build_target_observability_report(
        declaration, observability_report.observations,
    )
    if observability_report.to_dict() != expected_report.to_dict():
        raise ValueError("observability report disagrees with declaration")

    dates = declaration.forecast_dates
    horizon = len(dates)
    if tuple(row.target_date for row in alignment_result.rows) != dates:
        raise ValueError("alignment calendar mismatch")
    if (alignment_result.declared_horizon != horizon
            or metric_result.declared_horizon != horizon):
        raise ValueError("declared horizon mismatch")
    if alignment_result.coverage_mask != observability_report.coverage_mask:
        raise ValueError("alignment and observability coverage mismatch")
    if (alignment_result.eligible_count != observability_report.observed_count
            or alignment_result.excluded_count != observability_report.missing_count):
        raise ValueError("alignment and observability counts mismatch")

    for row, evidence in zip(alignment_result.rows, observability_report.observations):
        if row.observation_evidence != evidence:
            raise ValueError("alignment observation evidence mismatch")
        if row.eligible != evidence.verified_observed:
            raise ValueError("alignment eligibility mismatch")
        if row.eligible:
            if row.observed_microVolt != float(evidence.observation.value_microVolt):
                raise ValueError("alignment observed value mismatch")
            if row.exclusion_reasons:
                raise ValueError("eligible row has exclusion reasons")
        elif row.observed_microVolt is not None or not row.exclusion_reasons:
            raise ValueError("excluded row is inconsistent")

    if alignment_result.execution_artifact_digest != metric_result.execution_artifact_digest:
        raise ValueError("execution artifact digest mismatch")
    # Reuse the authoritative C2 computation, including zero-actual MAPE rules.
    expected_metrics = score_verified_observations(alignment_result)
    if metric_result.to_dict() != expected_metrics.to_dict():
        raise ValueError("observed metric result does not match alignment")

    return IntegratedAssessmentEvidence(
        contract_version=INTEGRATION_CONTRACT_VERSION,
        assessment_id=declaration.assessment_id,
        dataset_id=declaration.dataset_id,
        declaration_digest=declaration.digest,
        execution_artifact_digest=alignment_result.execution_artifact_digest,
        temporal_status=_temporal_status(temporal_evidence),
        coverage_status=observability_report.coverage_status.value,
        score_available=metric_result.score_available,
        temporal_evidence=temporal_evidence,
        observability_report=observability_report,
        alignment_result=alignment_result,
        metric_result=metric_result,
    )
