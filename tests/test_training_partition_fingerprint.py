"""E2-A2: read-only, fold-scoped provenance for canonical training partitions."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.fingerprint import (
    CANONICALIZATION_VERSION,
    FingerprintValidationError,
    fingerprint_dataset,
    fingerprint_training_partition,
    verify_training_partition_fingerprint,
)
from zenerestimation.optimization.fold import TemporalFold
from zenerestimation.optimization.validation import ExpandingWindowSplitter


def source_frame():
    return pd.DataFrame({
        "ds": pd.date_range("2022-03-01", periods=12, freq="QS-MAR"),
        "microVolt": [float(100 + i) for i in range(12)],
        "is_observed": [True] * 12,
    })


def make_fold(*, frame=None, folds=2, steps=2, index=0):
    data = source_frame() if frame is None else frame
    return ExpandingWindowSplitter(folds=folds, validation_steps=steps).split(
        BatteryDataset(data.copy(deep=True))
    )[index]


def test_fingerprint_has_expected_fields_and_canonical_digest():
    fold = make_fold()
    result = fingerprint_training_partition(fold)
    assert result.algorithm == "sha256"
    assert result.canonicalization_version == CANONICALIZATION_VERSION
    assert result.digest == fingerprint_dataset(fold.train.data).digest
    assert result.row_count == len(fold.train.data) == 8
    assert result.fold_index == 0
    assert result.train_start == "2022-03-01"
    assert result.train_end == "2023-12-01"
    assert result.validation_start == "2024-03-01"
    assert result.validation_end == "2024-06-01"


def test_determinism_and_dict():
    fold = make_fold()
    a = fingerprint_training_partition(fold)
    assert a == fingerprint_training_partition(fold)
    assert a.to_dict()["digest"] == a.digest
    assert verify_training_partition_fingerprint(fold, a)
    assert verify_training_partition_fingerprint(fold, a.to_dict())


def test_different_training_cutoff_changes_digest():
    first, second = ExpandingWindowSplitter(folds=2, validation_steps=2).split(
        BatteryDataset(source_frame())
    )
    assert fingerprint_training_partition(first).digest != fingerprint_training_partition(second).digest


def test_changed_training_value_changes_digest():
    frame = source_frame()
    original = fingerprint_training_partition(make_fold(frame=frame))
    frame.loc[2, "microVolt"] += 0.125
    assert fingerprint_training_partition(make_fold(frame=frame)).digest != original.digest


def test_validation_only_value_does_not_change_digest_or_metadata():
    fold = make_fold()
    before = fingerprint_training_partition(fold)
    fold.validation.loc[fold.validation.index[0], "microVolt"] += 10000
    assert fingerprint_training_partition(fold) == before


def test_future_source_value_does_not_change_training_digest():
    frame = source_frame()
    baseline = fingerprint_training_partition(make_fold(frame=frame))
    frame.loc[11, "microVolt"] += 10
    assert fingerprint_training_partition(make_fold(frame=frame)) == baseline


def test_fold_index_is_metadata_not_digest():
    fold = make_fold()
    clone = TemporalFold(index=9, train=fold.train, validation=fold.validation)
    a, b = fingerprint_training_partition(fold), fingerprint_training_partition(clone)
    assert a.digest == b.digest
    assert a.fold_index != b.fold_index
    assert not verify_training_partition_fingerprint(clone, a)


def test_validation_boundaries_are_metadata_not_digest():
    fold = make_fold()
    shorter = TemporalFold(index=fold.index, train=fold.train,
                           validation=fold.validation.iloc[:1].copy())
    a, b = fingerprint_training_partition(fold), fingerprint_training_partition(shorter)
    assert a.digest == b.digest
    assert a.validation_end != b.validation_end
    assert not verify_training_partition_fingerprint(shorter, a)


def test_train_and_validation_unchanged():
    fold = make_fold()
    train_before = fold.train.data.copy(deep=True)
    validation_before = fold.validation.copy(deep=True)
    fingerprint_training_partition(fold)
    pd.testing.assert_frame_equal(fold.train.data, train_before)
    pd.testing.assert_frame_equal(fold.validation, validation_before)


def test_inserted_missing_period_retained_in_hash():
    frame = source_frame()
    frame.loc[2, "microVolt"] = np.nan
    frame.loc[2, "is_observed"] = False
    fold = make_fold(frame=frame)
    result = fingerprint_training_partition(fold)
    assert result.digest == fingerprint_dataset(fold.train.data).digest
    no_missing_row = frame.drop(index=2).reset_index(drop=True)
    assert result.digest != fingerprint_training_partition(make_fold(frame=no_missing_row)).digest


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), "bad"])
def test_bad_training_measurement_rejected(bad):
    fold = make_fold()
    fold.train.data["microVolt"] = fold.train.data["microVolt"].astype(object)
    fold.train.data.loc[0, "microVolt"] = bad
    with pytest.raises(FingerprintValidationError):
        fingerprint_training_partition(fold)


def test_missing_observation_flags_rejected():
    fold = make_fold()
    fold.train.data.drop(columns="is_observed", inplace=True)
    with pytest.raises(FingerprintValidationError, match="is_observed"):
        fingerprint_training_partition(fold)


def test_invalid_inserted_training_value_rejected():
    fold = make_fold()
    fold.train.data.loc[0, "is_observed"] = False
    with pytest.raises(FingerprintValidationError, match="inserted period"):
        fingerprint_training_partition(fold)


def test_duplicate_training_date_rejected():
    fold = make_fold()
    fold.train.data.loc[2, "ds"] = fold.train.data.loc[1, "ds"]
    with pytest.raises(FingerprintValidationError, match="strictly increasing"):
        fingerprint_training_partition(fold)


def test_unsorted_training_dates_rejected_even_if_endpoints_valid():
    fold = make_fold()
    fold.train.data.loc[[1, 2], "ds"] = fold.train.data.loc[[2, 1], "ds"].to_numpy()
    with pytest.raises(FingerprintValidationError, match="strictly increasing"):
        fingerprint_training_partition(fold)


def test_unsorted_validation_dates_rejected():
    fold = make_fold()
    fold.validation.loc[fold.validation.index, "ds"] = fold.validation["ds"].iloc[::-1].to_numpy()
    with pytest.raises(FingerprintValidationError, match="strictly increasing"):
        fingerprint_training_partition(fold)


def test_duplicate_validation_dates_rejected():
    fold = make_fold()
    fold.validation.loc[fold.validation.index[1], "ds"] = fold.validation["ds"].iloc[0]
    with pytest.raises(FingerprintValidationError, match="strictly increasing"):
        fingerprint_training_partition(fold)


def test_boundary_overlap_rejected():
    fold = make_fold()
    fold.validation.loc[fold.validation.index[0], "ds"] = fold.train.data["ds"].iloc[-1]
    with pytest.raises(FingerprintValidationError, match="precede"):
        fingerprint_training_partition(fold)


def test_expected_boundary_change_returns_false():
    fold = make_fold()
    record = fingerprint_training_partition(fold).to_dict()
    record["train_end"] = "1999-01-01"
    assert not verify_training_partition_fingerprint(fold, record)


def test_expected_digest_change_returns_false():
    fold = make_fold()
    record = fingerprint_training_partition(fold).to_dict()
    record["digest"] = "0" * 64
    assert not verify_training_partition_fingerprint(fold, record)


def test_uppercase_digest_accepted():
    fold = make_fold()
    record = fingerprint_training_partition(fold).to_dict()
    record["digest"] = record["digest"].upper()
    assert verify_training_partition_fingerprint(fold, record)


@pytest.mark.parametrize("field,value", [
    ("algorithm", "md5"),
    ("canonicalization_version", "future-version"),
    ("digest", "not-a-sha256"),
])
def test_invalid_expected_contract_rejected(field, value):
    fold = make_fold()
    record = fingerprint_training_partition(fold).to_dict()
    record[field] = value
    with pytest.raises(FingerprintValidationError):
        verify_training_partition_fingerprint(fold, record)


def test_missing_expected_field_rejected():
    fold = make_fold()
    record = fingerprint_training_partition(fold).to_dict()
    del record["validation_end"]
    with pytest.raises(FingerprintValidationError, match="missing"):
        verify_training_partition_fingerprint(fold, record)


def test_wrong_fold_type_rejected():
    with pytest.raises(TypeError, match="TemporalFold"):
        fingerprint_training_partition(source_frame())


def test_wrong_expected_type_rejected():
    with pytest.raises(TypeError, match="mapping"):
        verify_training_partition_fingerprint(make_fold(), "bad")
