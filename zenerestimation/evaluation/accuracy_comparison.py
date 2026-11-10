
"""M17-C: read-only scientific comparison of measured B2 accuracy profiles."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import pandas as pd

from zenerestimation.evaluation.accuracy_diagnostics import (
    build_accuracy_diagnostics,
)
from zenerestimation.evaluation.accuracy_profile import (
    build_accuracy_profile,
)


_VERSION = "m17_b2_v1"
_COMPARISON_VERSION = "m17_c_v1"
_SCOPES = {"internal_expanding_window", "external_holdout"}


def _validate_profile(model: str, profile: Mapping) -> None:
    if not isinstance(profile, Mapping):
        raise TypeError("Each accuracy profile must be a mapping")

    if profile.get("profile_version") != _VERSION:
        raise ValueError("Unsupported accuracy profile version")

    provenance = profile.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Missing profile provenance")

    if provenance.get("model") != model:
        raise ValueError("Profile model identity mismatch")

    if not isinstance(provenance.get("dataset"), str) or not provenance["dataset"].strip():
        raise ValueError("Invalid dataset provenance")

    if provenance.get("scope") not in _SCOPES:
        raise ValueError("Invalid evaluation scope")

    if provenance.get("units") != "microVolt":
        raise ValueError("Invalid accuracy units")

    rows = profile.get("by_date")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing observation evidence")

    overall = profile.get("overall")
    if not isinstance(overall, Mapping):
        raise ValueError("Missing overall accuracy metrics")

    if overall.get("n") != len(rows):
        raise ValueError("Observation count mismatch")

    for field in ("rmse_microVolt", "mae_microVolt", "bias_microVolt"):
        value = overall.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid accuracy metric: {field}")

    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Invalid observation row")

        for field in (
            "fold_index",
            "horizon_step",
            "date",
            "actual_microVolt",
            "predicted_microVolt",
            "alignment_status",
        ):
            if field not in row:
                raise ValueError(f"Missing observation field: {field}")


def _observation_map(profile: Mapping) -> dict:
    observations = {}

    for row in profile["by_date"]:
        identity = (
            row["fold_index"],
            row["horizon_step"],
            row["date"],
        )

        if identity in observations:
            raise ValueError("Duplicate observation identity")

        observations[identity] = row["actual_microVolt"]

    return observations




def _same_evidence(
    left,
    right,
    *,
    path: tuple[str, ...] = (),
) -> bool:
    """Reconcile canonical evidence using field-specific equality rules."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (
            left.keys() == right.keys()
            and all(
                _same_evidence(
                    left[key],
                    right[key],
                    path=(*path, str(key)),
                )
                for key in left
            )
        )

    if isinstance(left, list) and isinstance(right, list):
        return (
            len(left) == len(right)
            and all(
                _same_evidence(
                    a,
                    b,
                    path=(*path, str(index)),
                )
                for index, (a, b) in enumerate(zip(left, right))
            )
        )

    # Booleans must never be accepted as numeric evidence.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right

    if isinstance(left, (int, float)) and isinstance(
        right, (int, float)
    ):
        if not (math.isfinite(left) and math.isfinite(right)):
            return False

        # Counts, identifiers, measurements and row-level errors
        # require exact numeric equality.
        metric_fields = {
            "rmse_microVolt",
            "mae_microVolt",
            "bias_microVolt",
            "mean_error_microVolt",
            "rmse_mean_microVolt",
            "rmse_std_population_microVolt",
            "worst_fold_rmse_microVolt",
        }

        aggregate_sections = {
            "overall",
            "first_step",
            "by_horizon",
            "fold_stability",
        }

        if (
            path
            and path[0] in aggregate_sections
            and path[-1] in metric_fields
        ):
            return math.isclose(
                left,
                right,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )

        return left == right

    return type(left) is type(right) and left == right



def _validate_canonical_profile(model: str, profile: Mapping) -> None:
    """Rebuild and reconcile an untrusted serialized B2 profile."""

    # Preserve the explicit duplicate-identity rejection contract.
    _observation_map(profile)

    rows = profile["by_date"]
    provenance = profile["provenance"]

    folds_by_index = {}

    for row in rows:
        fold_index = row["fold_index"]

        if (
            type(fold_index) is not int
            or fold_index < 0
        ):
            raise ValueError("Invalid fold index")

        folds_by_index.setdefault(fold_index, []).append(row)

    folds = []

    for fold_index, fold_rows in folds_by_index.items():
        try:
            dates = pd.to_datetime(
                [row["date"] for row in fold_rows],
                format="%Y-%m-%d",
                errors="raise",
            )

            actual = [
                row["actual_microVolt"]
                for row in fold_rows
            ]

            predicted = [
                row["predicted_microVolt"]
                for row in fold_rows
            ]

            alignment = {
                row["alignment_status"]
                for row in fold_rows
            }

            if len(alignment) != 1:
                raise ValueError(
                    "Mixed alignment statuses within fold"
                )

            folds.append(
                {
                    "fold_index": fold_index,
                    "dates": dates,
                    "actual": actual,
                    "predicted": predicted,
                    "alignment_status": next(iter(alignment)),
                }
            )

        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Invalid observation evidence"
            ) from exc

    try:
        diagnostics = build_accuracy_diagnostics(
            folds,
            dataset=provenance["dataset"],
            model=model,
            scope=provenance["scope"],
        )

        canonical = build_accuracy_profile(diagnostics)

    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Cannot reconstruct canonical accuracy profile"
        ) from exc

    if not _same_evidence(profile, canonical):
        raise ValueError(
            "Accuracy profile does not reconcile "
            "with canonical measured evidence"
        )



def compare_accuracy_profiles(
    profiles: Mapping[str, Mapping],
    *,
    comparison_policy: str = "strict",
) -> dict:
    """Compare saved B2 profiles without forecasting or model selection."""

    if not isinstance(profiles, Mapping):
        raise TypeError("profiles must be a mapping")

    if len(profiles) < 2:
        raise ValueError("At least two accuracy profiles are required")

    if comparison_policy != "strict":
        raise ValueError("Unsupported comparison policy")

    for model, profile in profiles.items():
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Invalid model identity")

        _validate_profile(model, profile)
        _validate_canonical_profile(model, profile)

    models = list(profiles)

    # Use a deterministic reference for scientific compatibility checks.
    comparison_models = sorted(models)

    reference = profiles[comparison_models[0]]
    reference_observations = _observation_map(reference)

    reasons = []

    if not reference["alignment_summary"]["fully_verified"]:
        reasons.append("Reference profile lacks verified alignment")

    for model in comparison_models[1:]:
        candidate = profiles[model]
        candidate_observations = _observation_map(candidate)

        reference_provenance = reference["provenance"]
        candidate_provenance = candidate["provenance"]

        if reference_provenance["dataset"] != candidate_provenance["dataset"]:
            reasons.append(f"{model}: dataset mismatch")

        if reference_provenance["scope"] != candidate_provenance["scope"]:
            reasons.append(f"{model}: evaluation scope mismatch")

        if reference_observations.keys() != candidate_observations.keys():
            reasons.append(f"{model}: observation identities mismatch")
        else:
            for identity in reference_observations:
                actual_a = reference_observations[identity]
                actual_b = candidate_observations[identity]

                if actual_a != actual_b:
                    raise ValueError(
                        "Conflicting measurement for matched observation"
                    )

        if not candidate["alignment_summary"]["fully_verified"]:
            reasons.append(f"{model}: alignment is not fully verified")

    return {
        "comparison_version": _COMPARISON_VERSION,
        "comparison_policy": comparison_policy,
        "models": models,
        "profiles": deepcopy(dict(profiles)),
        "comparability": {
            "status": (
                "directly_comparable"
                if not reasons
                else "descriptive_only"
            ),
            "reasons": reasons,
        },
        "ranking": None,
    }
