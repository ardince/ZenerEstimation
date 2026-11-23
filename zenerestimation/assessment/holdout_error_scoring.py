"""M19-B3-B3: deterministic observed-error scoring after controlled disclosure.

This scores only already disclosed, declaration-bound targets. It does not run
models, read the holdout partition, authenticate evidence, or prove independence.
Error convention matches M19-B2-I2-C2: forecast minus observed.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import fsum, isfinite, sqrt
from collections.abc import Mapping

from zenerestimation.assessment.eligibility import AssessmentDeclaration, validate_assessment_declaration
from zenerestimation.assessment.holdout_target_disclosure import (
    DISCLOSURE_VERSION, _check_eligibility, _canonical, _digest,
)
from zenerestimation.assessment.observability import OBSERVABILITY_CONTRACT_VERSION
from zenerestimation.assessment.observed_metrics import MAPE_ZERO_POLICY

SCORING_VERSION = "m19_b3_b3_holdout_error_scoring_v1"
_DISCLOSURE_KEYS = frozenset({
    "disclosure_version", "status", "assessment_id", "declaration_digest",
    "dataset_id", "eligibility_digest", "execution_artifact_digest",
    "isolation_version", "observability_contract_version", "forecast_dates",
    "forecast_horizon", "observed_microVolt", "observation_evidence",
    "scientific_independence", "metrics_included", "disclosure_digest",
})
_EVIDENCE_KEYS = frozenset({
    "observation", "provenance", "availability", "exposure", "exposure_status", "verified_observed",
})
_OBSERVATION_KEYS = frozenset({
    "dataset_id", "target_date", "value_microVolt", "origin", "source_ref", "availability_date",
})


def score_disclosed_holdout(
    *, declaration: AssessmentDeclaration, eligibility_snapshot: Mapping,
    disclosure_snapshot: Mapping,
) -> dict:
    """Revalidate B1/B2 snapshots and score exact declared date pairs.

    Digest verification is internal consistency, not source authentication.
    Inputs must originate from the approved, access-controlled B1/B2 workflow.
    """
    validate_assessment_declaration(declaration)
    approved = _check_eligibility(eligibility_snapshot, declaration)
    if type(disclosure_snapshot) is not dict or set(disclosure_snapshot) != _DISCLOSURE_KEYS:
        raise ValueError("B2 disclosure schema mismatch")
    try:
        disclosed = json.loads(_canonical(disclosure_snapshot))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("B2 disclosure is not canonical JSON") from exc
    if disclosed != disclosure_snapshot:
        raise ValueError("B2 disclosure has noncanonical types")
    supplied_digest = disclosed.pop("disclosure_digest")
    if type(supplied_digest) is not str or supplied_digest != _digest(disclosed):
        raise ValueError("B2 disclosure digest mismatch")
    expected = {
        "disclosure_version": DISCLOSURE_VERSION,
        "status": "TARGETS_DISCLOSED_FOR_EVALUATION",
        "assessment_id": declaration.assessment_id,
        "declaration_digest": declaration.digest,
        "dataset_id": declaration.dataset_id,
        "eligibility_digest": eligibility_snapshot["eligibility_digest"],
        "execution_artifact_digest": approved["execution_artifact_digest"],
        "observability_contract_version": OBSERVABILITY_CONTRACT_VERSION,
        "forecast_dates": list(declaration.forecast_dates),
        "forecast_horizon": len(declaration.forecast_dates),
        "scientific_independence": "not_established",
        "metrics_included": False,
    }
    for key, value in expected.items():
        if type(disclosed[key]) is not type(value) or disclosed[key] != value:
            raise ValueError(f"B2 disclosure {key} mismatch")
    # Isolation version is a required provenance marker, not proof of isolation.
    from zenerestimation.assessment.holdout_isolation import ISOLATION_VERSION
    if disclosed["isolation_version"] != ISOLATION_VERSION:
        raise ValueError("B2 disclosure isolation version mismatch")

    dates = list(declaration.forecast_dates)
    actuals = disclosed["observed_microVolt"]
    predictions = approved["predicted_microVolt"]
    evidence = disclosed["observation_evidence"]
    n = len(dates)
    if (type(actuals) is not list or len(actuals) != n or
            type(evidence) is not list or len(evidence) != n):
        raise ValueError("B2 disclosure observation horizon mismatch")
    rows = []
    errors = []
    percentages = []
    zero_dates = []
    for date, prediction, actual, item in zip(dates, predictions, actuals, evidence):
        if type(actual) not in (int, float) or not isfinite(actual):
            raise ValueError("B2 disclosure nonfinite target")
        if type(item) is not dict or set(item) != _EVIDENCE_KEYS:
            raise ValueError("B2 observation evidence schema mismatch")
        observation = item["observation"]
        if type(observation) is not dict or set(observation) != _OBSERVATION_KEYS:
            raise ValueError("B2 observation schema mismatch")
        if (observation["dataset_id"] != declaration.dataset_id or
                observation["target_date"] != date or
                type(observation["value_microVolt"]) not in (int, float) or
                not isfinite(observation["value_microVolt"]) or
                float(observation["value_microVolt"]) != float(actual) or
                observation["origin"] != "measured" or
                type(observation["source_ref"]) is not str or
                not observation["source_ref"] or item["verified_observed"] is not True):
            raise ValueError("B2 observation is not aligned verified measurement")
        provenance = item["provenance"]
        if type(provenance) is not dict or provenance.get("status") != "verified":
            raise ValueError("B2 observation provenance not verified")
        error = float(prediction) - float(actual)
        if not isfinite(error):
            raise ValueError("nonfinite forecast error")
        errors.append(error)
        rows.append({
            "target_date": date, "forecast_microVolt": float(prediction),
            "observed_microVolt": float(actual), "signed_error_microVolt": error,
            "absolute_error_microVolt": abs(error),
            "squared_error_microVolt2": error * error,
        })
        if actual == 0:
            zero_dates.append(date)
        else:
            percentage = abs(error / actual) * 100.0
            if not isfinite(percentage):
                raise ValueError("nonfinite percentage error")
            percentages.append(percentage)
    rmse = sqrt(fsum(e * e for e in errors) / n)
    mae = fsum(abs(e) for e in errors) / n
    bias = fsum(errors) / n
    mape = fsum(percentages) / len(percentages) if percentages else None
    if not all(isfinite(v) for v in (rmse, mae, bias)):
        raise ValueError("nonfinite aggregate metric")
    if any(not isfinite(row["squared_error_microVolt2"]) for row in rows):
        raise ValueError("nonfinite squared error")
    payload = {
        "scoring_version": SCORING_VERSION,
        "status": "HOLDOUT_SCORED",
        "assessment_id": declaration.assessment_id,
        "declaration_digest": declaration.digest,
        "dataset_id": declaration.dataset_id,
        "eligibility_digest": eligibility_snapshot["eligibility_digest"],
        "disclosure_digest": supplied_digest,
        "execution_artifact_digest": approved["execution_artifact_digest"],
        "forecast_dates": dates,
        "forecast_horizon": n,
        "scored_count": n,
        "error_convention": "forecast_minus_observed",
        "metrics": {"rmse_microVolt": rmse, "mae_microVolt": mae,
                    "bias_microVolt": bias, "mape_percent": mape},
        "denominators": {"rmse": n, "mae": n, "bias": n, "mape": len(percentages)},
        "mape_zero_policy": MAPE_ZERO_POLICY,
        "mape_zero_actual_excluded_dates": zero_dates,
        "by_date": rows,
        "scientific_independence": "not_established",
    }
    result = json.loads(_canonical(payload))
    result["scoring_digest"] = _digest(result)
    return result
