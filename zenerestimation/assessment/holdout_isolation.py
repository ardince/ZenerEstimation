"""M19-B3-A2: fail-closed partition isolation for declared holdout dates.

This establishes *data-partition* consistency only. It does not prove that
model selection or historical training was free of target exposure.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date

import pandas as pd

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, validate_assessment_declaration,
)
from zenerestimation.data.dataset import BatteryDataset

ISOLATION_VERSION = "m19_b3_a4_date_defined_isolation_v1"


@dataclass(frozen=True)
class IsolatedHoldout:
    development: BatteryDataset
    holdout: pd.DataFrame
    evidence: dict


def _canonical_dates(frame: pd.DataFrame) -> list[str]:
    if "ds" not in frame or frame.empty:
        raise ValueError("dataset requires nonempty ds column")
    parsed = pd.to_datetime(frame["ds"], errors="raise")
    if parsed.isna().any():
        raise ValueError("null dataset timestamp")
    if any(v.tzinfo is not None or v != v.normalize() for v in parsed):
        raise ValueError("dataset timestamps must be naive calendar dates")
    return [v.strftime("%Y-%m-%d") for v in parsed]


def isolate_declared_holdout(
    dataset: BatteryDataset, *, declaration: AssessmentDeclaration,
) -> IsolatedHoldout:
    """Partition by declared cutoff and exact holdout calendar, not row count.

    No optimization, fitting, prediction, or scoring occurs. Returns detached
    partitions. The caller must pass `development`, never `dataset`, to any
    optimizer or training procedure.
    """
    validate_assessment_declaration(declaration)
    if not isinstance(dataset, BatteryDataset):
        raise TypeError("dataset must be BatteryDataset")
    source = dataset.data.copy(deep=True)
    dates = _canonical_dates(source)
    if len(set(dates)) != len(dates):
        raise ValueError("duplicate dataset timestamps")
    if "microVolt" not in source:
        raise ValueError("dataset requires microVolt measurements")
    if len(source) <= len(declaration.forecast_dates):
        raise ValueError("no development observations remain")

    # Partition by the declared temporal boundary, never by the last N rows.
    # Validate the entire future calendar before exposing any training data.
    ordered = source.assign(_canonical_ds=dates).sort_values(
        "_canonical_ds", kind="stable"
    ).reset_index(drop=True)
    cutoff = declaration.development_cutoff
    expected_holdout = list(declaration.forecast_dates)
    development_rows = ordered.loc[ordered["_canonical_ds"] <= cutoff]
    holdout_rows = ordered.loc[ordered["_canonical_ds"] > cutoff]
    development_dates = development_rows["_canonical_ds"].tolist()
    holdout_dates = holdout_rows["_canonical_ds"].tolist()

    if holdout_dates != expected_holdout:
        raise ValueError("holdout calendar does not match declaration")
    if not development_dates or development_dates[-1] != cutoff:
        raise ValueError("development cutoff does not match partition")
    if any(date.fromisoformat(v) <= date.fromisoformat(cutoff)
           for v in holdout_dates):
        raise ValueError("holdout overlaps development")

    development_frame = development_rows.drop(columns="_canonical_ds").copy(deep=True).reset_index(drop=True)
    holdout = holdout_rows.drop(columns="_canonical_ds").copy(deep=True).reset_index(drop=True)
    development = deepcopy(dataset)
    development.data = development_frame
    if any(pd.isna(v) for v in development.data["microVolt"]):
        raise ValueError("development contains missing measurements")

    return IsolatedHoldout(
        development=development,
        holdout=holdout,
        evidence={
            "isolation_version": ISOLATION_VERSION,
            "status": "PARTITION_VERIFIED",
            "assessment_id": declaration.assessment_id,
            "declaration_digest": declaration.digest,
            "dataset_id": declaration.dataset_id,
            "development_cutoff": development_dates[-1],
            "development_rows": len(development_dates),
            "holdout_dates": holdout_dates.copy(),
            "holdout_rows": len(holdout_dates),
            "scientific_independence": "not_established",
        },
    )


def require_isolated_training_dataset(
    candidate: BatteryDataset, *, isolated: IsolatedHoldout,
) -> None:
    """Reject full-dataset or otherwise altered training inputs at a call site."""
    if not isinstance(candidate, BatteryDataset):
        raise TypeError("candidate must be BatteryDataset")
    if not isinstance(isolated, IsolatedHoldout):
        raise TypeError("isolated must be IsolatedHoldout")
    expected = isolated.development.data.reset_index(drop=True)
    actual = candidate.data.reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=True)
    except AssertionError as exc:
        raise ValueError("training input differs from isolated development partition") from exc
