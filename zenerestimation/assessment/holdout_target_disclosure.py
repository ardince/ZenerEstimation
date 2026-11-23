"""M19-B3-B2: controlled disclosure of measured holdout targets.

Requires a digest-consistent B1 snapshot, a verified A4 partition record,
and complete measured/provenance-verified B2 observation evidence. This is
structural consistency, not external provenance authentication or freeze proof.
No model execution or scoring occurs.
"""
from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from math import isfinite

import pandas as pd

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, validate_assessment_declaration,
)
from zenerestimation.assessment.holdout_evaluation_eligibility import ELIGIBILITY_VERSION
from zenerestimation.assessment.holdout_isolation import IsolatedHoldout, ISOLATION_VERSION
from zenerestimation.assessment.observability import (
    TargetObservabilityReport, build_target_observability_report,
)

DISCLOSURE_VERSION = "m19_b3_b2_controlled_target_disclosure_v1"
_ELIGIBILITY_KEYS = frozenset({
    "eligibility_version", "status", "assessment_id", "declaration_digest",
    "dataset_id", "development_cutoff", "selected_model", "decision_digest",
    "plan_digest", "execution_artifact_digest", "training_fingerprint",
    "forecast_horizon", "forecast_dates", "predicted_microVolt",
    "scientific_independence", "target_values_included", "eligibility_digest",
})
_ISOLATION_KEYS = frozenset({
    "isolation_version", "status", "assessment_id", "declaration_digest",
    "dataset_id", "development_cutoff", "development_rows", "holdout_dates",
    "holdout_rows", "scientific_independence",
})


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + sha256(_canonical(value)).hexdigest()


def _check_eligibility(snapshot: Mapping, declaration: AssessmentDeclaration) -> dict:
    if type(snapshot) is not dict or set(snapshot) != _ELIGIBILITY_KEYS:
        raise ValueError("B1 eligibility snapshot schema mismatch")
    try:
        detached = json.loads(_canonical(snapshot))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("B1 eligibility snapshot is not canonical JSON") from exc
    if detached != snapshot:
        raise ValueError("B1 eligibility snapshot has noncanonical types")
    digest = detached.pop("eligibility_digest")
    if type(digest) is not str or digest != _digest(detached):
        raise ValueError("B1 eligibility digest mismatch")
    expected = {
        "eligibility_version": ELIGIBILITY_VERSION,
        "status": "ELIGIBLE_FOR_HOLDOUT_DISCLOSURE",
        "assessment_id": declaration.assessment_id,
        "declaration_digest": declaration.digest,
        "dataset_id": declaration.dataset_id,
        "development_cutoff": declaration.development_cutoff,
        "forecast_horizon": len(declaration.forecast_dates),
        "forecast_dates": list(declaration.forecast_dates),
        "scientific_independence": "not_established",
        "target_values_included": False,
    }
    for key, value in expected.items():
        if type(detached[key]) is not type(value) or detached[key] != value:
            raise ValueError(f"B1 eligibility {key} mismatch")
    values = detached["predicted_microVolt"]
    if type(values) is not list or len(values) != len(declaration.forecast_dates) or any(
        type(v) not in (int, float) or not isfinite(v) for v in values
    ):
        raise ValueError("B1 eligibility predictions invalid")
    if type(detached["selected_model"]) is not str or not detached["selected_model"]:
        raise ValueError("B1 eligibility model invalid")
    for key in ("decision_digest", "plan_digest", "execution_artifact_digest"):
        if type(detached[key]) is not str or not detached[key]:
            raise ValueError(f"B1 eligibility {key} invalid")
    if type(detached["training_fingerprint"]) is not dict:
        raise ValueError("B1 eligibility training fingerprint invalid")
    return detached


def disclose_holdout_targets(
    *, declaration: AssessmentDeclaration, eligibility_snapshot: Mapping,
    isolated: IsolatedHoldout, observability_report: TargetObservabilityReport,
) -> dict:
    """Disclose aligned measured targets only after B1 and observation checks.

    Caller is responsible for protecting the disclosure output from training
    and selection code. A B1 digest alone does not authenticate its origin.
    """
    validate_assessment_declaration(declaration)
    approved = _check_eligibility(eligibility_snapshot, declaration)
    if not isinstance(isolated, IsolatedHoldout):
        raise TypeError("isolated must be IsolatedHoldout")
    evidence = isolated.evidence
    if type(evidence) is not dict or set(evidence) != _ISOLATION_KEYS:
        raise ValueError("isolation evidence schema mismatch")
    expected = {
        "isolation_version": ISOLATION_VERSION,
        "status": "PARTITION_VERIFIED",
        "assessment_id": declaration.assessment_id,
        "declaration_digest": declaration.digest,
        "dataset_id": declaration.dataset_id,
        "development_cutoff": declaration.development_cutoff,
        "holdout_dates": list(declaration.forecast_dates),
        "holdout_rows": len(declaration.forecast_dates),
        "scientific_independence": "not_established",
    }
    for key, value in expected.items():
        if type(evidence[key]) is not type(value) or evidence[key] != value:
            raise ValueError(f"isolation {key} mismatch")
    if type(evidence["development_rows"]) is not int or evidence["development_rows"] <= 0:
        raise ValueError("isolation development rows invalid")
    if not isinstance(isolated.holdout, pd.DataFrame):
        raise TypeError("isolated holdout must be a DataFrame")
    frame = isolated.holdout
    if len(frame) != len(declaration.forecast_dates) or "ds" not in frame or "microVolt" not in frame:
        raise ValueError("holdout rows or required columns mismatch")
    dates = pd.to_datetime(frame["ds"], errors="raise")
    if dates.isna().any() or any(v.tzinfo is not None or v != v.normalize() for v in dates):
        raise ValueError("holdout timestamps invalid")
    if [v.strftime("%Y-%m-%d") for v in dates] != list(declaration.forecast_dates):
        raise ValueError("holdout calendar mismatch")
    values = frame["microVolt"].tolist()
    if any(type(v) not in (int, float) or not isfinite(v) for v in values):
        raise ValueError("holdout targets must be finite numeric scalars")

    if not isinstance(observability_report, TargetObservabilityReport):
        raise TypeError("observability_report must be TargetObservabilityReport")
    if (observability_report.assessment_id != declaration.assessment_id
            or observability_report.dataset_id != declaration.dataset_id
            or observability_report.declaration_digest != declaration.digest):
        raise ValueError("observability declaration binding mismatch")
    # Existing B2 machinery checks exact declared calendar and coverage.
    checked = build_target_observability_report(declaration, observability_report.observations)
    if checked != observability_report or not all(checked.coverage_mask):
        raise ValueError("complete verified measured observation coverage required")
    for index, entry in enumerate(checked.observations):
        observation = entry.observation
        if float(observation.value_microVolt) != float(values[index]):
            raise ValueError("observability target differs from isolated holdout")

    payload = {
        "disclosure_version": DISCLOSURE_VERSION,
        "status": "TARGETS_DISCLOSED_FOR_EVALUATION",
        "assessment_id": declaration.assessment_id,
        "declaration_digest": declaration.digest,
        "dataset_id": declaration.dataset_id,
        "eligibility_digest": eligibility_snapshot["eligibility_digest"],
        "execution_artifact_digest": approved["execution_artifact_digest"],
        "isolation_version": ISOLATION_VERSION,
        "observability_contract_version": checked.contract_version,
        "forecast_dates": list(declaration.forecast_dates),
        "forecast_horizon": len(declaration.forecast_dates),
        "observed_microVolt": [float(v) for v in values],
        "observation_evidence": [entry.to_dict() for entry in checked.observations],
        "scientific_independence": "not_established",
        "metrics_included": False,
    }
    detached = json.loads(_canonical(payload))
    detached["disclosure_digest"] = _digest(detached)
    return detached
