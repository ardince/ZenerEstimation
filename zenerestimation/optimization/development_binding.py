"""M20-B3-C: development-only binding for locally approved optimization.

Consistency checks, not authentication of source provenance or scientific independence.
The caller must supply a trusted source dataset and approved M19 declaration.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

import pandas as pd

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, validate_assessment_declaration,
)
from zenerestimation.assessment.holdout_isolation import (
    ISOLATION_VERSION, isolate_declared_holdout, require_isolated_training_dataset,
)
from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.fingerprint import fingerprint_dataset
from zenerestimation.optimization.authorization import OptimizationExecutionRequest
from zenerestimation.optimization.internal_provenance import (
    PROTOCOL_VERSION, build_internal_validation_context,
)
from zenerestimation.optimization.validation import ExpandingWindowSplitter

BINDING_VERSION = "m20_b3_development_binding_v1"
VALIDATION_DIGEST_VERSION = "m20_b3_validation_protocol_v1"
_PROTOCOL_KEYS = (
    "protocol_version", "scope", "folds", "validation_steps",
    "fold_validation_dates", "fold_training_end_dates",
)


def build_validation_protocol_digest(internal_context: dict[str, Any]) -> str:
    """Hash the actual M18 fold calendar and configuration, not dataset targets."""
    if type(internal_context) is not dict:
        raise TypeError("internal_context must be a dictionary")
    if internal_context.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported internal provenance protocol")
    if internal_context.get("scope") != "internal_expanding_window":
        raise ValueError("invalid internal validation scope")
    if any(key not in internal_context for key in _PROTOCOL_KEYS):
        raise ValueError("incomplete validation protocol")
    payload = {
        "digest_version": VALIDATION_DIGEST_VERSION,
        **{key: internal_context[key] for key in _PROTOCOL_KEYS},
    }
    # Reject malformed/non-JSON protocol evidence rather than stringify it.
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedDevelopmentBinding:
    """Detached snapshot; mutable internals require revalidation before execution."""
    development: BatteryDataset
    evidence: dict[str, Any]


def verify_authorized_development_dataset(
    request: OptimizationExecutionRequest,
    *,
    declaration: AssessmentDeclaration,
    source_dataset: BatteryDataset,
    training_dataset: BatteryDataset,
    splitter: ExpandingWindowSplitter,
) -> VerifiedDevelopmentBinding:
    """Repartition trusted source and verify exact authorized development input.

    Never returns holdout values. No grant consumption, fitting or optimization.
    The trusted caller is responsible for binding the approved declaration and
    source to the approval workflow; a request alone cannot authenticate them.
    """
    if type(request) is not OptimizationExecutionRequest:
        raise TypeError("request must be OptimizationExecutionRequest")
    validate_assessment_declaration(declaration)
    if not isinstance(source_dataset, BatteryDataset):
        raise TypeError("source_dataset must be BatteryDataset")
    if not isinstance(training_dataset, BatteryDataset):
        raise TypeError("training_dataset must be BatteryDataset")
    if not isinstance(splitter, ExpandingWindowSplitter):
        raise TypeError("splitter must be ExpandingWindowSplitter")
    if request.dataset_id != declaration.dataset_id:
        raise ValueError("authorization dataset_id mismatch")
    if request.development_cutoff != declaration.development_cutoff:
        raise ValueError("authorization development cutoff mismatch")

    # Work exclusively on detached inputs to prevent caller mutation mid-check.
    source_copy = deepcopy(source_dataset)
    training_copy = deepcopy(training_dataset)
    isolated = isolate_declared_holdout(source_copy, declaration=declaration)
    require_isolated_training_dataset(training_copy, isolated=isolated)

    evidence = isolated.evidence
    expected_evidence = {
        "isolation_version": ISOLATION_VERSION,
        "status": "PARTITION_VERIFIED",
        "assessment_id": declaration.assessment_id,
        "declaration_digest": declaration.digest,
        "dataset_id": declaration.dataset_id,
        "development_cutoff": declaration.development_cutoff,
        "development_rows": len(isolated.development.data),
        "holdout_dates": list(declaration.forecast_dates),
        "holdout_rows": len(declaration.forecast_dates),
        "scientific_independence": "not_established",
    }
    if evidence != expected_evidence:
        raise ValueError("isolation evidence inconsistent with declaration")

    # v1 requires explicit observation flags; never silently infer their value.
    # The processed fingerprint is supplementary to the M18 authorization hash.
    processed = fingerprint_dataset(training_copy.data)
    context = build_internal_validation_context(
        training_copy, splitter, dataset_id=request.dataset_id,
    )
    if context["dataset_fingerprint"] != request.development_fingerprint:
        raise ValueError("authorized development fingerprint mismatch")
    protocol_digest = build_validation_protocol_digest(context)
    if protocol_digest != request.validation_protocol_digest:
        raise ValueError("authorized validation protocol mismatch")
    actual_cutoff = pd.Timestamp(training_copy.data["ds"].iloc[-1]).strftime("%Y-%m-%d")
    if actual_cutoff != request.development_cutoff:
        raise ValueError("actual development cutoff mismatch")

    # Deliberately omit holdout targets, holdout DataFrame, and full source.
    record = {
        "binding_version": BINDING_VERSION,
        "status": "DEVELOPMENT_BOUND",
        "request_digest": request.digest,
        "declaration_digest": declaration.digest,
        "dataset_id": request.dataset_id,
        "development_cutoff": actual_cutoff,
        "development_fingerprint": context["dataset_fingerprint"],
        "processed_dataset_fingerprint": processed.to_dict(),
        "validation_protocol_digest": protocol_digest,
        "validation_protocol_version": VALIDATION_DIGEST_VERSION,
        "isolation_version": ISOLATION_VERSION,
        "development_rows": len(training_copy.data),
        "scientific_independence": "not_established",
        "source_authenticity": "not_established",
    }
    return VerifiedDevelopmentBinding(development=deepcopy(training_copy), evidence=record)
