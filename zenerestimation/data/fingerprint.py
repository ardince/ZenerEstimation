"""Versioned, read-only fingerprints for processed battery datasets.

Canonical format v1 identifies ds, microVolt and is_observed only. It does
not hash raw/legacy datasets, interpolated training inputs, or arbitrary
preprocessing state. Those require separate, explicit contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

CANONICALIZATION_VERSION = "zener-canonical-dataset-v1"
ALGORITHM = "sha256"
_COLUMNS = ("ds", "microVolt", "is_observed")


class FingerprintValidationError(ValueError):
    """Input cannot be represented unambiguously under the v1 contract."""


@dataclass(frozen=True)
class DatasetFingerprint:
    algorithm: str
    digest: str
    canonicalization_version: str
    row_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "digest": self.digest,
            "canonicalization_version": self.canonicalization_version,
            "row_count": self.row_count,
        }

    def to_manifest_fingerprint(self) -> dict[str, str]:
        """Schema 1.0-compatible subset; retain version separately."""
        return {"algorithm": self.algorithm, "digest": self.digest}


def fingerprint_file(path: str | Path) -> str:
    """SHA-256 of exact file bytes, streamed without changing the file."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame(dataset: Any) -> pd.DataFrame:
    # Import lazily to avoid changing module import behavior.
    from zenerestimation.data.dataset import BatteryDataset

    if isinstance(dataset, BatteryDataset):
        if not dataset.is_processed:
            raise FingerprintValidationError(
                "v1 requires a processed BatteryDataset; raw/legacy datasets are unsupported"
            )
        return dataset.data.copy(deep=True)
    if isinstance(dataset, pd.DataFrame):
        return dataset.copy(deep=True)
    raise TypeError("dataset must be a processed BatteryDataset or pandas DataFrame")


def _canonical_rows(dataset: Any) -> list[list[Any]]:
    frame = _frame(dataset)
    if frame.empty:
        raise FingerprintValidationError("dataset must not be empty")
    missing = set(_COLUMNS) - set(frame.columns)
    if missing:
        raise FingerprintValidationError(f"missing required columns: {sorted(missing)}")

    # Explicit observation flags are mandatory; never infer them.
    if not frame["is_observed"].map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise FingerprintValidationError("is_observed must contain explicit booleans")

    rows: list[tuple[str, str | None, bool]] = []
    for index, (date, value, observed) in enumerate(
        frame.loc[:, _COLUMNS].itertuples(index=False, name=None)
    ):
        if not isinstance(date, (pd.Timestamp, datetime, np.datetime64)):
            raise FingerprintValidationError(f"row {index}: ds must be datetime")
        try:
            stamp = pd.Timestamp(date)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FingerprintValidationError(f"row {index}: invalid ds") from exc
        if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
            raise FingerprintValidationError(
                f"row {index}: ds must be a timezone-naive calendar date"
            )
        date_text = stamp.strftime("%Y-%m-%d")
        flag = bool(observed)
        if value is None or (isinstance(value, (float, np.floating)) and math.isnan(value)) or value is pd.NA:
            if flag:
                raise FingerprintValidationError(f"row {index}: observed value is missing")
            encoded_value = None
        else:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise FingerprintValidationError(f"row {index}: microVolt must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise FingerprintValidationError(f"row {index}: nonfinite microVolt")
            if not flag:
                raise FingerprintValidationError(
                    f"row {index}: inserted period must have missing microVolt"
                )
            # Canonicalize signed zero; use exact binary64 round-trip representation.
            encoded_value = (0.0 if number == 0.0 else number).hex()
        rows.append((date_text, encoded_value, flag))

    rows.sort(key=lambda row: row[0])
    dates = [row[0] for row in rows]
    if len(set(dates)) != len(dates):
        raise FingerprintValidationError("duplicate timestamps")
    return [list(row) for row in rows]


def fingerprint_dataset(dataset: Any) -> DatasetFingerprint:
    """Hash canonical processed data without mutating the input."""
    rows = _canonical_rows(dataset)
    payload = {
        "canonicalization_version": CANONICALIZATION_VERSION,
        "columns": list(_COLUMNS),
        "rows": rows,
    }
    canonical_bytes = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return DatasetFingerprint(
        algorithm=ALGORITHM,
        digest=sha256(canonical_bytes).hexdigest(),
        canonicalization_version=CANONICALIZATION_VERSION,
        row_count=len(rows),
    )


def verify_dataset_fingerprint(
    dataset: Any, expected: DatasetFingerprint | Mapping[str, Any]
) -> bool:
    """Return True for a matching versioned record; reject incomplete metadata."""
    if isinstance(expected, DatasetFingerprint):
        expected = expected.to_dict()
    if not isinstance(expected, Mapping):
        raise TypeError("expected must be a DatasetFingerprint or mapping")
    required = {"algorithm", "digest", "canonicalization_version"}
    if not required.issubset(expected):
        raise FingerprintValidationError("expected fingerprint requires algorithm, digest and canonicalization_version")
    if expected["algorithm"] != ALGORITHM:
        raise FingerprintValidationError("unsupported fingerprint algorithm")
    if expected["canonicalization_version"] != CANONICALIZATION_VERSION:
        raise FingerprintValidationError("unsupported canonicalization version")
    digest = expected["digest"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
        raise FingerprintValidationError("expected digest must be 64 hex characters")
    actual = fingerprint_dataset(dataset)
    if "row_count" in expected and expected["row_count"] != actual.row_count:
        return False
    return actual.digest == digest.lower()


# E2-A2: fold-level provenance. The data digest intentionally remains the
# E2-A1 canonical digest; validation targets never enter the hashed payload.
@dataclass(frozen=True)
class TrainingPartitionFingerprint:
    algorithm: str
    digest: str
    canonicalization_version: str
    row_count: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    fold_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "digest": self.digest,
            "canonicalization_version": self.canonicalization_version,
            "row_count": self.row_count,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "fold_index": self.fold_index,
        }


def _ordered_partition_dates(frame: pd.DataFrame, name: str) -> list[str]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.empty or "ds" not in frame.columns:
        raise FingerprintValidationError(f"{name} requires nonempty ds column")
    dates: list[str] = []
    previous: pd.Timestamp | None = None
    for index, value in enumerate(frame["ds"]):
        if not isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
            raise FingerprintValidationError(f"{name} row {index}: ds must be datetime")
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise FingerprintValidationError(f"{name} row {index}: invalid ds") from exc
        if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
            raise FingerprintValidationError(
                f"{name} row {index}: ds must be a timezone-naive calendar date"
            )
        if previous is not None and stamp <= previous:
            raise FingerprintValidationError(
                f"{name} timestamps must be strictly increasing without duplicates"
            )
        dates.append(stamp.strftime("%Y-%m-%d"))
        previous = stamp
    return dates


def fingerprint_training_partition(fold: Any) -> TrainingPartitionFingerprint:
    """Fingerprint an actual TemporalFold, before any training interpolation.

    The canonical digest covers training data only; the other fields provide
    temporal provenance. No preprocessing, sorting, or mutation is performed.
    """
    from zenerestimation.optimization.fold import TemporalFold

    if not isinstance(fold, TemporalFold):
        raise TypeError("fold must be a TemporalFold")
    train = fold.train.data.copy(deep=True)
    validation = fold.validation.copy(deep=True)
    train_dates = _ordered_partition_dates(train, "train")
    validation_dates = _ordered_partition_dates(validation, "validation")
    if train_dates[-1] >= validation_dates[0]:
        raise FingerprintValidationError(
            "training observations must strictly precede validation observations"
        )
    # _canonical_rows enforces explicit observed flags and the original
    # processed-data missing-value contract. Pass DataFrame because splitter
    # creates a new BatteryDataset without preserving is_processed metadata.
    data_fingerprint = fingerprint_dataset(train)
    return TrainingPartitionFingerprint(
        algorithm=data_fingerprint.algorithm,
        digest=data_fingerprint.digest,
        canonicalization_version=data_fingerprint.canonicalization_version,
        row_count=data_fingerprint.row_count,
        train_start=train_dates[0],
        train_end=train_dates[-1],
        validation_start=validation_dates[0],
        validation_end=validation_dates[-1],
        fold_index=fold.index,
    )


def verify_training_partition_fingerprint(
    fold: Any, expected: TrainingPartitionFingerprint | Mapping[str, Any]
) -> bool:
    """Verify both training content and fold boundary/index provenance."""
    if isinstance(expected, TrainingPartitionFingerprint):
        expected = expected.to_dict()
    if not isinstance(expected, Mapping):
        raise TypeError("expected must be a TrainingPartitionFingerprint or mapping")
    fields = set(TrainingPartitionFingerprint.__dataclass_fields__)
    if not fields.issubset(expected):
        raise FingerprintValidationError(
            f"expected partition fingerprint missing: {sorted(fields - set(expected))}"
        )
    if expected["algorithm"] != ALGORITHM:
        raise FingerprintValidationError("unsupported fingerprint algorithm")
    if expected["canonicalization_version"] != CANONICALIZATION_VERSION:
        raise FingerprintValidationError("unsupported canonicalization version")
    digest = expected["digest"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in digest)):
        raise FingerprintValidationError("expected digest must be 64 hex characters")
    actual = fingerprint_training_partition(fold).to_dict()
    return all(
        actual[field] == (digest.lower() if field == "digest" else expected[field])
        for field in fields
    )
