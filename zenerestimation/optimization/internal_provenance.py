
"""M18-B2a: read-only provenance for internal expanding-window validation.

A fingerprint identifies the supplied development data, not the dataset used
by an earlier model run. Linking that evidence is the responsibility of B2b.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.optimization.validation import ExpandingWindowSplitter


PROTOCOL_VERSION = "m18_internal_provenance_v1"


def _canonical_dataset(
    dataset: BatteryDataset,
) -> tuple[list[dict], pd.DatetimeIndex]:
    if not isinstance(dataset, BatteryDataset):
        raise TypeError("dataset must be a BatteryDataset")

    data = dataset.data

    if not {"ds", "microVolt"}.issubset(data.columns):
        raise ValueError("dataset requires ds and microVolt columns")

    if data.empty:
        raise ValueError("development dataset cannot be empty")

    # Preserve the row order used by the evaluator.
    if not pd.api.types.is_datetime64_any_dtype(data["ds"]):
        raise ValueError("ds must contain parsed datetime timestamps")

    dates = pd.DatetimeIndex(data["ds"])

    if (
        dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
    ):
        raise ValueError(
            "development timestamps must be nonmissing, unique and increasing"
        )

    if dates.tz is not None or not all(
        date == date.normalize() for date in dates
    ):
        raise ValueError(
            "development timestamps must be timezone-naive calendar dates"
        )

    if "is_observed" in data:
        observed = data["is_observed"]

        if not all(
            isinstance(value, (bool, np.bool_))
            for value in observed
        ):
            raise ValueError(
                "is_observed must contain only booleans"
            )

        flags = [bool(value) for value in observed]
    else:
        flags = [True] * len(data)

    values = pd.to_numeric(
        data["microVolt"],
        errors="raise",
    )

    records = []

    for date, raw, flag in zip(dates, values, flags):
        value = float(raw)

        if math.isinf(value):
            raise ValueError(
                "development targets cannot be infinite"
            )

        if flag and not math.isfinite(value):
            raise ValueError(
                "observed measurements must be finite"
            )

        if not flag and not math.isnan(value):
            raise ValueError(
                "inserted periods must have missing targets"
            )

        records.append({
            "ds": date.strftime("%Y-%m-%d"),
            "microVolt": (
                None if math.isnan(value) else value.hex()
            ),
            "is_observed": flag,
        })

    return records, dates


def build_internal_validation_context(
    dataset: BatteryDataset,
    splitter: ExpandingWindowSplitter,
    *,
    dataset_id: str,
) -> dict:
    """Describe the development dataset and expanding-window protocol.

    Returns JSON-compatible evidence without modifying the inputs,
    fitting models, or accessing external holdout observations.
    """
    if not isinstance(splitter, ExpandingWindowSplitter):
        raise TypeError(
            "splitter must be an ExpandingWindowSplitter"
        )

    if (
        not isinstance(dataset_id, str)
        or not dataset_id.strip()
    ):
        raise ValueError(
            "dataset_id must be a nonempty string"
        )

    records, dates = _canonical_dataset(dataset)

    folds = splitter.split(dataset)

    fold_dates = []
    training_ends = []

    for fold in folds:
        validation = fold.validation

        val_dates = pd.DatetimeIndex(
            validation["ds"]
        )

        if (
            "is_observed" in validation
            and not validation["is_observed"].all()
        ):
            raise ValueError(
                "validation folds cannot contain inserted periods"
            )

        actual = pd.to_numeric(
            validation["microVolt"],
            errors="raise",
        ).to_numpy(dtype=float)

        if not np.isfinite(actual).all():
            raise ValueError(
                "validation folds require finite measured targets"
            )

        fold_dates.append([
            date.strftime("%Y-%m-%d")
            for date in val_dates
        ])

        training_ends.append(
            pd.Timestamp(
                fold.train.data["ds"].iloc[-1]
            ).strftime("%Y-%m-%d")
        )

    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    fingerprint = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_id": dataset_id,
        "dataset_fingerprint": f"sha256:{fingerprint}",
        "scope": "internal_expanding_window",
        "fold_validation_dates": fold_dates,
        "fold_training_end_dates": training_ends,
        "folds": splitter.folds,
        "validation_steps": splitter.validation_steps,
    }
