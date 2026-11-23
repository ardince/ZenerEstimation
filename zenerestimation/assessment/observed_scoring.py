"""M19-B2-I2-C1: declaration-bound alignment, not accuracy scoring.

Evidence is caller supplied. Verification of an M18 execution artifact establishes
consistency with trusted inputs, not independent proof that a model ran.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, validate_assessment_declaration,
)
from zenerestimation.assessment.observability import (
    TargetObservabilityReport, TargetObservationEvidence,
    build_target_observability_report,
)
from zenerestimation.optimization.internal_forecast_execution_artifact import (
    verify_internal_forecast_execution_artifact,
)

ALIGNMENT_CONTRACT_VERSION = "M19_B2_I2_C1_ALIGNMENT_V1"


@dataclass(frozen=True, slots=True)
class ObservedAlignmentRow:
    """One declared forecast quarter, whether or not it can be scored later."""

    target_date: str
    forecast_microVolt: float
    observed_microVolt: float | None
    eligible: bool
    exclusion_reasons: tuple[str, ...]
    observation_evidence: TargetObservationEvidence

    def __post_init__(self) -> None:
        if type(self.forecast_microVolt) not in (float, int) or not isfinite(self.forecast_microVolt):
            raise ValueError("forecast_microVolt must be finite numeric")
        if self.eligible != (self.observed_microVolt is not None):
            raise ValueError("eligible rows must have observed values, excluded rows must not")
        if self.eligible and self.exclusion_reasons:
            raise ValueError("eligible rows cannot have exclusion reasons")
        if not self.eligible and not self.exclusion_reasons:
            raise ValueError("excluded rows require reasons")
        if self.target_date != self.observation_evidence.observation.target_date:
            raise ValueError("row date does not match observation evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "target_date": self.target_date,
            "forecast_microVolt": self.forecast_microVolt,
            "observed_microVolt": self.observed_microVolt,
            "eligible": self.eligible,
            "exclusion_reasons": list(self.exclusion_reasons),
            "observation_evidence": self.observation_evidence.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ObservedAlignmentResult:
    contract_version: str
    assessment_id: str
    dataset_id: str
    declaration_digest: str
    execution_artifact_digest: str
    rows: tuple[ObservedAlignmentRow, ...]

    @property
    def declared_horizon(self) -> int:
        return len(self.rows)

    @property
    def eligible_count(self) -> int:
        return sum(row.eligible for row in self.rows)

    @property
    def excluded_count(self) -> int:
        return self.declared_horizon - self.eligible_count

    @property
    def coverage_mask(self) -> tuple[bool, ...]:
        return tuple(row.eligible for row in self.rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "assessment_id": self.assessment_id,
            "dataset_id": self.dataset_id,
            "declaration_digest": self.declaration_digest,
            "execution_artifact_digest": self.execution_artifact_digest,
            "declared_horizon": self.declared_horizon,
            "eligible_count": self.eligible_count,
            "excluded_count": self.excluded_count,
            "coverage_mask": list(self.coverage_mask),
            "rows": [row.to_dict() for row in self.rows],
        }


def _exclusion_reasons(item: TargetObservationEvidence) -> tuple[str, ...]:
    if item.verified_observed:
        return ()
    reasons = []
    if item.observation.origin.value != "measured":
        reasons.append("ORIGIN_NOT_MEASURED")
    if item.provenance.status.value != "verified":
        reasons.append("PROVENANCE_NOT_VERIFIED")
    return tuple(reasons)


def align_verified_observations(
    *,
    declaration: AssessmentDeclaration,
    observability_report: TargetObservabilityReport,
    execution_artifact: Mapping,
    plan: Mapping,
    artifact: Mapping,
    results: Mapping,
    validation_context: Mapping,
    registry,
    training_dataset,
    training_context: Mapping,
) -> ObservedAlignmentResult:
    """Align a verified M18 forecast to supplied M19 observation evidence.

    Fail closed on mismatched calendars and execution evidence. Do not infer
    observation provenance, availability, exposure, or scientific independence.
    """
    validate_assessment_declaration(declaration)
    if not isinstance(observability_report, TargetObservabilityReport):
        raise TypeError("observability_report must be TargetObservabilityReport")
    if (observability_report.assessment_id != declaration.assessment_id
            or observability_report.dataset_id != declaration.dataset_id
            or observability_report.declaration_digest != declaration.digest):
        raise ValueError("observability report declaration binding mismatch")
    if observability_report.declared_horizon != len(declaration.forecast_dates):
        raise ValueError("observability report horizon mismatch")
    # The report constructor alone checks sorted dates, not the declaration's calendar.
    build_target_observability_report(declaration, observability_report.observations)

    if not isinstance(plan, Mapping):
        raise TypeError("plan must be a mapping")
    dates = list(declaration.forecast_dates)
    horizon = len(dates)
    if type(plan.get("forecast_horizon")) is not int or plan["forecast_horizon"] != horizon:
        raise ValueError("plan horizon does not match declaration")
    if type(plan.get("forecast_dates")) is not list or plan["forecast_dates"] != dates:
        raise ValueError("plan dates do not match declaration")
    if not isinstance(execution_artifact, Mapping):
        raise TypeError("execution_artifact must be a mapping")
    if not verify_internal_forecast_execution_artifact(
        execution_artifact, plan=plan, artifact=artifact, results=results,
        validation_context=validation_context, registry=registry,
        training_dataset=training_dataset, training_context=training_context,
        forecast_horizon=horizon, forecast_dates=dates,
    ):
        raise ValueError("execution artifact verification failed")

    # Read forecast only after verifier success; do not accept an alternate forecast.
    try:
        forecast = execution_artifact["execution_record"]["forecast"]
        values = forecast["values_microVolt"]
        if (type(forecast["horizon"]) is not int or forecast["horizon"] != horizon
                or type(forecast["dates"]) is not list or forecast["dates"] != dates
                or type(values) is not list or len(values) != horizon
                or any(type(v) not in (int, float) or not isfinite(v) for v in values)):
            raise ValueError("execution forecast does not match declared horizon")
        digest = execution_artifact["artifact_digest"]
        if type(digest) is not str:
            raise ValueError("execution artifact digest missing")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("malformed verified execution forecast") from exc

    rows = []
    for target_date, prediction, evidence in zip(dates, values, observability_report.observations):
        eligible = evidence.verified_observed
        observed = float(evidence.observation.value_microVolt) if eligible else None
        rows.append(ObservedAlignmentRow(
            target_date=target_date,
            forecast_microVolt=float(prediction),
            observed_microVolt=observed,
            eligible=eligible,
            exclusion_reasons=_exclusion_reasons(evidence),
            observation_evidence=evidence,
        ))
    return ObservedAlignmentResult(
        contract_version=ALIGNMENT_CONTRACT_VERSION,
        assessment_id=declaration.assessment_id,
        dataset_id=declaration.dataset_id,
        declaration_digest=declaration.digest,
        execution_artifact_digest=digest,
        rows=tuple(rows),
    )
