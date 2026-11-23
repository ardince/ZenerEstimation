"""M19-B2-I2-A: read-only temporal eligibility evidence.

This module verifies supplied M18 artifacts against independently supplied M18
inputs. It cannot authenticate development history, freeze times, or target
availability. Findings about those subjects remain conditional on their source.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, VerificationFinding, VerificationStatus,
    validate_assessment_declaration, validate_verification_finding,
)
from zenerestimation.optimization.internal_forecast_plan import verify_internal_forecast_plan
from zenerestimation.optimization.internal_forecast_execution_artifact import (
    verify_internal_forecast_execution_artifact,
)

TEMPORAL_CONTRACT_VERSION = "M19_B2_I2_A_TEMPORAL_V1"
_VERIFIER = "m19_i2_a_temporal_policy"


def _identifier(value: str, field: str) -> None:
    if type(value) is not str or not value or not all(
        c.isascii() and (c.isalnum() or c in "._:-") for c in value
    ) or not value[0].isalnum():
        raise ValueError(f"{field} must be a canonical identifier")


def _finding(status: VerificationStatus, reason: str | None = None,
             evidence_ref: str | None = None) -> VerificationFinding:
    return VerificationFinding(
        status=status,
        evidence_ref=evidence_ref,
        verifier_id=_VERIFIER if status is VerificationStatus.VERIFIED else None,
        reason_codes=(reason,) if reason else (),
    )


@dataclass(frozen=True, slots=True)
class TemporalEligibilityRequest:
    declaration: AssessmentDeclaration
    plan_ref: str
    execution_artifact_ref: str
    development_evidence_ref: str

    def __post_init__(self) -> None:
        validate_assessment_declaration(self.declaration)
        for field in ("plan_ref", "execution_artifact_ref", "development_evidence_ref"):
            _identifier(getattr(self, field), field)
        # If I1 declares a corresponding binding, it must not contradict I2.
        for key, attr in (("plan", "plan_ref"),
                          ("execution_artifact", "execution_artifact_ref"),
                          ("development", "development_evidence_ref")):
            expected = self.declaration.evidence_refs.get(key)
            if expected is not None and expected != getattr(self, attr):
                raise ValueError(f"declaration evidence_refs[{key}] mismatch")


@dataclass(frozen=True, slots=True)
class TemporalEligibilityEvidence:
    assessment_id: str
    declaration_digest: str
    plan_binding: VerificationFinding
    execution_integrity: VerificationFinding
    training_cutoff: VerificationFinding
    development_boundary: VerificationFinding
    forecast_freeze: VerificationFinding
    target_availability: VerificationFinding
    chronology: VerificationFinding
    contract_version: str = TEMPORAL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _identifier(self.assessment_id, "assessment_id")
        if (type(self.declaration_digest) is not str or
            not self.declaration_digest.startswith("sha256:") or
            len(self.declaration_digest) != 71 or
            any(c not in "0123456789abcdef" for c in self.declaration_digest[7:])):
            raise ValueError("invalid declaration digest")
        if self.contract_version != TEMPORAL_CONTRACT_VERSION:
            raise ValueError("unsupported temporal contract version")
        for name in ("plan_binding", "execution_integrity", "training_cutoff",
                     "development_boundary", "forecast_freeze", "target_availability",
                     "chronology"):
            validate_verification_finding(getattr(self, name))

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "assessment_id": self.assessment_id,
            "declaration_digest": self.declaration_digest,
            **{name: getattr(self, name).to_dict() for name in (
                "plan_binding", "execution_integrity", "training_cutoff",
                "development_boundary", "forecast_freeze", "target_availability",
                "chronology")},
        }


def evaluate_temporal_eligibility(
    request: TemporalEligibilityRequest,
    *,
    plan: Mapping[str, Any],
    execution_artifact: Mapping[str, Any],
    selection_artifact: Mapping[str, Any],
    results: Mapping[str, Any],
    validation_context: Mapping[str, Any],
    registry: Any,
    training_dataset: Any,
    training_context: Mapping[str, Any],
    forecast_horizon: int,
    forecast_dates: Any,
    development_boundary_evidence: VerificationFinding,
    forecast_freeze_evidence: VerificationFinding,
    target_availability_evidence: VerificationFinding,
    chronology_evidence: VerificationFinding,
) -> TemporalEligibilityEvidence:
    """Return conditional temporal findings; do not execute a forecasting model.

    M18 verifiers are authoritative *only relative to trusted caller inputs*.
    Supplied chronology/development findings are not authenticated here.
    Malformed M18 trusted inputs may raise, matching the M18 verifier contract.
    """
    if not isinstance(request, TemporalEligibilityRequest):
        raise TypeError("request must be TemporalEligibilityRequest")
    for name, finding in (
        ("development_boundary_evidence", development_boundary_evidence),
        ("forecast_freeze_evidence", forecast_freeze_evidence),
        ("target_availability_evidence", target_availability_evidence),
        ("chronology_evidence", chronology_evidence),
    ):
        if not isinstance(finding, VerificationFinding):
            raise TypeError(f"{name} must be VerificationFinding")
        validate_verification_finding(finding)

    declaration = request.declaration
    shared = dict(
        artifact=selection_artifact, results=results,
        validation_context=validation_context, registry=registry,
        training_dataset=training_dataset, training_context=training_context,
        forecast_horizon=forecast_horizon, forecast_dates=forecast_dates,
    )
    plan_ok = verify_internal_forecast_plan(plan, **shared)
    # Do not use unverified plan fields to establish binding or cutoff.
    plan_binding = _finding(VerificationStatus.FAILED, "PLAN_VERIFICATION_FAILED")
    training_cutoff = _finding(VerificationStatus.UNKNOWN, "TRAINING_CUTOFF_UNVERIFIED")
    execution_integrity = _finding(VerificationStatus.UNKNOWN, "EXECUTION_UNVERIFIED")
    if plan_ok:
        calendar_ok = (
            type(plan.get("forecast_horizon")) is int
            and plan["forecast_horizon"] == len(declaration.forecast_dates)
            and type(plan.get("forecast_dates")) is list
            and plan["forecast_dates"] == list(declaration.forecast_dates)
            and type(forecast_horizon) is int
            and forecast_horizon == len(declaration.forecast_dates)
        )
        plan_binding = (_finding(VerificationStatus.VERIFIED, evidence_ref=request.plan_ref)
                        if calendar_ok else
                        _finding(VerificationStatus.FAILED, "ASSESSMENT_PLAN_MISMATCH"))
        provenance = plan["training_provenance"]
        endpoint = date.fromisoformat(provenance["train_end"])
        cutoff = date.fromisoformat(declaration.development_cutoff)
        training_cutoff = (
            _finding(VerificationStatus.VERIFIED, evidence_ref=request.plan_ref)
            if endpoint <= cutoff else
            _finding(VerificationStatus.FAILED, "TRAINING_AFTER_DEVELOPMENT_CUTOFF")
        )
        artifact_ok = verify_internal_forecast_execution_artifact(
            execution_artifact, plan=plan, **shared,
        )
        execution_integrity = (
            _finding(VerificationStatus.VERIFIED, evidence_ref=request.execution_artifact_ref)
            if artifact_ok else
            _finding(VerificationStatus.FAILED, "EXECUTION_ARTIFACT_VERIFICATION_FAILED")
        )
    return TemporalEligibilityEvidence(
        assessment_id=declaration.assessment_id,
        declaration_digest=declaration.digest,
        plan_binding=plan_binding,
        execution_integrity=execution_integrity,
        training_cutoff=training_cutoff,
        development_boundary=development_boundary_evidence,
        forecast_freeze=forecast_freeze_evidence,
        target_availability=target_availability_evidence,
        chronology=chronology_evidence,
    )
