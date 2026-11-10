
"""M17-C4: read-only presentation of scientific RMSE acceptance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from zenerestimation.evaluation.accuracy_comparison import (
    compare_accuracy_profiles,
)
from zenerestimation.evaluation.accuracy_acceptance import (
    evaluate_accuracy_threshold,
)


_COLUMNS = [
    "model",
    "rmse_microVolt",
    "threshold_microVolt",
    "status",
    "comparison_status",
]


def _validate_selected_models(
    profiles: Mapping,
    models: Sequence[str],
) -> list[str]:
    """Validate presentation selection without changing evidence."""

    if isinstance(models, (str, bytes)) or not isinstance(
        models, Sequence
    ):
        raise TypeError("models must be a sequence of model names")

    selected = list(models)

    if not selected:
        raise ValueError("Select at least one model")

    if any(
        not isinstance(model, str) or not model.strip()
        for model in selected
    ):
        raise ValueError("Invalid selected model identity")

    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate selected models")

    unknown = [
        model for model in selected
        if model not in profiles
    ]

    if unknown:
        raise ValueError(
            "Unknown selected models: " + ", ".join(unknown)
        )

    return selected


def build_accuracy_acceptance_view(
    profiles: Mapping,
    *,
    models: Sequence[str],
    threshold_microVolt: float = 0.5,
) -> pd.DataFrame:
    """Build a read-only acceptance table from complete B2 evidence.

    All supplied profiles participate in scientific comparability,
    regardless of which models are selected for presentation.

    No model fitting, forecasting, optimization, ranking, or
    automatic selection occurs.
    """

    if not isinstance(profiles, Mapping):
        raise TypeError("profiles must be a mapping")

    selected = _validate_selected_models(
        profiles,
        models,
    )

    # Validate the complete evidence collection, not just
    # the models currently selected for display.
    comparison = compare_accuracy_profiles(profiles)

    # C3 is the sole authority for RMSE threshold decisions.
    acceptance = evaluate_accuracy_threshold(
        comparison,
        metric="overall_rmse",
        threshold_microVolt=threshold_microVolt,
    )

    rows = []

    for model in selected:
        assessment = acceptance["acceptance"][model]

        rows.append(
            {
                "model": model,
                "rmse_microVolt": assessment[
                    "rmse_microVolt"
                ],
                "threshold_microVolt": acceptance[
                    "threshold_microVolt"
                ],
                "status": assessment["status"],
                "comparison_status": acceptance[
                    "comparison_status"
                ],
            }
        )

    return pd.DataFrame(
        rows,
        columns=_COLUMNS,
    )


def build_acceptance_signature(
    profiles: Mapping,
    *,
    models: Sequence[str],
    threshold_microVolt: float,
) -> tuple:
    """Identify the exact evidence and UI inputs approved for display."""

    import hashlib
    import json
    import math

    if not isinstance(profiles, Mapping):
        raise TypeError("profiles must be a mapping")

    selected = _validate_selected_models(profiles, models)

    if (
        isinstance(threshold_microVolt, bool)
        or not isinstance(threshold_microVolt, (int, float))
    ):
        raise TypeError("Invalid acceptance threshold")

    threshold = float(threshold_microVolt)

    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid acceptance threshold")

    evidence_json = json.dumps(
        profiles,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    evidence_digest = hashlib.sha256(
        evidence_json.encode("utf-8")
    ).hexdigest()

    return (
        evidence_digest,
        tuple(selected),
        threshold,
    )
