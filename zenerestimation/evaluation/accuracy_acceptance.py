
"""M17-C3: read-only acceptance of validated measured accuracy profiles."""

from __future__ import annotations

from collections.abc import Mapping
import math

from zenerestimation.evaluation.accuracy_comparison import (
    compare_accuracy_profiles,
)


_ACCEPTANCE_VERSION = "m17_c3_v1"
_SUPPORTED_METRICS = {
    "overall_rmse": ("overall", "rmse_microVolt"),
}


def _validate_threshold(value: float) -> float:
    """Validate an inclusive, finite, nonnegative RMSE threshold."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError(
            "threshold_microVolt must be a finite numeric value"
        )

    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "threshold_microVolt must be finite and nonnegative"
        )

    return float(value)


def _validate_comparison(comparison: Mapping) -> dict:
    """Rebuild the comparison rather than trusting serialized claims."""
    if not isinstance(comparison, Mapping):
        raise TypeError("comparison must be a mapping")

    profiles = comparison.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("Missing comparison profiles")

    policy = comparison.get("comparison_policy")
    if policy != "strict":
        raise ValueError("Unsupported comparison policy")

    # Revalidate every embedded B2 profile and all comparability
    # evidence through the M17-C2 scientific contract.
    canonical = compare_accuracy_profiles(
        profiles,
        comparison_policy=policy,
    )

    # The report must match the reconstructed C2 report exactly.
    # This rejects altered status, reasons, model order, version,
    # ranking, missing fields and unexpected extra fields.
    if dict(comparison) != canonical:
        raise ValueError(
            "Comparison report does not reconcile "
            "with canonical M17-C2 evidence"
        )

    return canonical


def evaluate_accuracy_threshold(
    comparison: Mapping,
    *,
    metric: str = "overall_rmse",
    threshold_microVolt: float = 0.5,
) -> dict:
    """Apply a read-only RMSE acceptance criterion.

    This function does not fit, forecast, optimize, rank or select
    models. It only assesses previously measured and validated
    M17-B2 accuracy evidence.

    Descriptive-only comparisons receive 'not_eligible' rather
    than a comparative pass/fail decision.
    """
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(f"Unsupported acceptance metric: {metric}")

    threshold = _validate_threshold(threshold_microVolt)
    canonical = _validate_comparison(comparison)

    comparison_status = canonical["comparability"]["status"]
    section, field = _SUPPORTED_METRICS[metric]

    acceptance = {}

    for model in canonical["models"]:
        profile = canonical["profiles"][model]
        rmse = profile[section][field]

        if comparison_status != "directly_comparable":
            status = "not_eligible"
        elif rmse <= threshold:
            status = "pass"
        else:
            status = "fail"

        acceptance[model] = {
            "rmse_microVolt": rmse,
            "status": status,
        }

    return {
        "acceptance_version": _ACCEPTANCE_VERSION,
        "metric": metric,
        "threshold_microVolt": threshold,
        "comparison_status": comparison_status,
        "acceptance": acceptance,
        "ranking": None,
    }
