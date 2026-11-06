
"""Complete measurement-based accuracy profile for M17-B2.

B1 remains the numerical authority. This module validates B1
evidence before constructing a separate, immutable-by-copy profile.

No model fitting, forecasting, candidate ranking, or holdout-based
optimization is performed here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from math import isclose, isfinite

import numpy as np
import pandas as pd

from zenerestimation.evaluation.accuracy_diagnostics import (
    build_accuracy_diagnostics,
)


_PROFILE_VERSION = "m17_b2_v1"

_ALLOWED_SCOPES = frozenset({
    "internal_expanding_window",
    "external_holdout",
})

_ALLOWED_ALIGNMENTS = frozenset({
    "verified",
    "positional_only",
})

_REQUIRED_ROW_FIELDS = frozenset({
    "fold_index",
    "horizon_step",
    "date",
    "actual_microVolt",
    "predicted_microVolt",
    "error_microVolt",
    "absolute_error_microVolt",
    "squared_error_microVolt2",
    "alignment_status",
})

_REQUIRED_REPORT_FIELDS = frozenset({
    "rows",
    "summary",
    "horizon_metrics",
    "fold_metrics",
    "provenance",
})


def _require_number(value, name: str) -> float:
    """Validate a finite, non-boolean numeric value."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite numeric data")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be finite numeric data"
        ) from exc

    if not isfinite(number):
        raise ValueError(f"{name} must be finite")

    return number


def _require_integer(value, name: str, minimum: int) -> int:
    """Validate an integer without accepting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")

    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")

    return int(value)


def _equal_evidence(actual, expected, path="diagnostics") -> None:
    """Compare supplied evidence with canonical B1 evidence."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError(f"{path} must be a mapping")

        if set(actual) != set(expected):
            raise ValueError(f"{path} has inconsistent fields")

        for key, value in expected.items():
            _equal_evidence(
                actual[key],
                value,
                f"{path}.{key}",
            )
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise ValueError(f"{path} must be a list")

        if len(actual) != len(expected):
            raise ValueError(f"{path} has inconsistent length")

        for index, (left, right) in enumerate(
            zip(actual, expected)
        ):
            _equal_evidence(
                left,
                right,
                f"{path}[{index}]",
            )
        return

    if isinstance(expected, bool):
        if not isinstance(actual, (bool, np.bool_)):
            raise ValueError(f"{path} has inconsistent type")
        if bool(actual) != expected:
            raise ValueError(f"{path} is inconsistent")
        return

    if isinstance(expected, int):
        if _require_integer(actual, path, 0) != expected:
            raise ValueError(f"{path} is inconsistent")
        return

    if isinstance(expected, float):
        value = _require_number(actual, path)

        if not isclose(
            value,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{path} is inconsistent")
        return

    if actual != expected:
        raise ValueError(f"{path} is inconsistent")


def _reconstruct_b1(diagnostics: Mapping) -> dict:
    """Reconstruct canonical B1 evidence from recorded observations."""
    if not isinstance(diagnostics, Mapping):
        raise ValueError("diagnostics must be a mapping")

    missing = _REQUIRED_REPORT_FIELDS - set(diagnostics)
    if missing:
        raise ValueError(
            f"Missing diagnostics fields: {sorted(missing)}"
        )

    provenance = diagnostics["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")

    scope = provenance.get("scope")
    if scope not in _ALLOWED_SCOPES:
        raise ValueError(
            "scope must identify a measured evaluation"
        )

    rows = diagnostics["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "rows must contain measured observations"
        )

    grouped = defaultdict(list)
    identities = set()

    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Each row must be a mapping")

        missing_fields = _REQUIRED_ROW_FIELDS - set(row)
        if missing_fields:
            raise ValueError(
                f"Missing observation fields: {sorted(missing_fields)}"
            )

        fold_index = _require_integer(
            row["fold_index"], "fold_index", 0
        )
        horizon_step = _require_integer(
            row["horizon_step"], "horizon_step", 1
        )

        identity = (fold_index, horizon_step)
        if identity in identities:
            raise ValueError(
                "Duplicate observation identity"
            )
        identities.add(identity)

        date_text = row["date"]
        if not isinstance(date_text, str):
            raise ValueError("date must be YYYY-MM-DD")

        try:
            date = datetime.strptime(
                date_text, "%Y-%m-%d"
            )
        except ValueError as exc:
            raise ValueError(
                "Invalid observation date"
            ) from exc

        if date.strftime("%Y-%m-%d") != date_text:
            raise ValueError("Noncanonical observation date")

        actual = _require_number(
            row["actual_microVolt"],
            "actual_microVolt",
        )
        predicted = _require_number(
            row["predicted_microVolt"],
            "predicted_microVolt",
        )

        error = predicted - actual
        if not isfinite(error) or not isfinite(error * error):
            raise ValueError(
                "Error magnitude exceeds finite numeric range"
            )

        for field in (
            "error_microVolt",
            "absolute_error_microVolt",
            "squared_error_microVolt2",
        ):
            _require_number(row[field], field)

        status = row["alignment_status"]
        if status not in _ALLOWED_ALIGNMENTS:
            raise ValueError("Unknown alignment status")

        grouped[fold_index].append(
            (horizon_step, date, actual, predicted, status)
        )

    folds = []

    for fold_index, observations in grouped.items():
        # Preserve the recorded order. A reordered or missing
        # horizon must not be silently repaired.
        steps = [item[0] for item in observations]
        if steps != list(range(1, len(steps) + 1)):
            raise ValueError(
                "Fold horizon steps must be consecutive"
            )

        statuses = {item[4] for item in observations}
        if len(statuses) != 1:
            raise ValueError(
                "Alignment status must be consistent within a fold"
            )

        dates = pd.DatetimeIndex(
            [item[1] for item in observations]
        )
        if not dates.is_monotonic_increasing or not dates.is_unique:
            raise ValueError(
                "Fold dates must be unique and increasing"
            )

        folds.append({
            "fold_index": fold_index,
            "dates": dates,
            "actual": np.array(
                [item[2] for item in observations],
                dtype=float,
            ),
            "predicted": np.array(
                [item[3] for item in observations],
                dtype=float,
            ),
            "alignment_status": observations[0][4],
        })

    return build_accuracy_diagnostics(
        folds,
        dataset=provenance.get("dataset"),
        model=provenance.get("model"),
        scope=scope,
    )


def build_accuracy_profile(diagnostics: Mapping) -> dict:
    """Build a verified accuracy profile from a B1 report.

    Rejects inconsistent observation evidence, aggregate metrics,
    fold metrics, horizon metrics, and provenance.
    """
    canonical = _reconstruct_b1(diagnostics)

    # Validate the complete report, including alignment provenance,
    # observation identities, and all numerical aggregations.
    _equal_evidence(diagnostics, canonical)

    rows = canonical["rows"]
    summary = canonical["summary"]

    first_errors = np.asarray(
        [
            row["error_microVolt"]
            for row in rows
            if row["horizon_step"] == 1
        ],
        dtype=float,
    )

    first_step = {
        "n": int(first_errors.size),
        "mean_error_microVolt": float(
            np.mean(first_errors)
        ),
        "mae_microVolt": float(
            np.mean(np.abs(first_errors))
        ),
        "rmse_microVolt": float(
            np.sqrt(np.mean(np.square(first_errors)))
        ),
    }

    verified_rows = sum(
        row["alignment_status"] == "verified"
        for row in rows
    )
    positional_rows = sum(
        row["alignment_status"] == "positional_only"
        for row in rows
    )

    return {
        "profile_version": _PROFILE_VERSION,
        "provenance": deepcopy(canonical["provenance"]),
        "overall": {
            key: summary[key]
            for key in (
                "n",
                "rmse_microVolt",
                "mae_microVolt",
                "bias_microVolt",
            )
        },
        "first_step": first_step,
        "by_horizon": deepcopy(
            canonical["horizon_metrics"]
        ),
        "by_date": deepcopy(rows),
        "fold_stability": deepcopy(
            summary["fold_stability"]
        ),
        "alignment_summary": {
            "total_rows": len(rows),
            "verified_rows": verified_rows,
            "positional_only_rows": positional_rows,
            "fully_verified": verified_rows == len(rows),
        },
    }
