"""E2-A1: processed-dataset fingerprint contract, no training or repo changes."""
import copy
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.fingerprint import (
    CANONICALIZATION_VERSION,
    DatasetFingerprint,
    FingerprintValidationError,
    fingerprint_dataset,
    fingerprint_file,
    verify_dataset_fingerprint,
)


@pytest.fixture
def frame():
    return pd.DataFrame({
        'ds': pd.to_datetime(['2023-03-01', '2023-06-01', '2023-09-01']),
        'microVolt': [30.0, np.nan, 30.5],
        'is_observed': [True, False, True],
    })


def test_repeatable_and_structured(frame):
    first = fingerprint_dataset(frame)
    assert first == fingerprint_dataset(frame)
    assert first.algorithm == 'sha256'
    assert first.canonicalization_version == CANONICALIZATION_VERSION
    assert first.row_count == 3
    assert len(first.digest) == 64
    assert first.to_manifest_fingerprint() == {'algorithm': 'sha256', 'digest': first.digest}


def test_row_order_invariant(frame):
    assert fingerprint_dataset(frame) == fingerprint_dataset(frame.iloc[::-1].reset_index(drop=True))


def test_column_order_and_extras_ignored(frame):
    changed = frame[['is_observed', 'ds', 'microVolt']].copy()
    changed['delta'] = [100, 200, 300]
    assert fingerprint_dataset(frame) == fingerprint_dataset(changed)


def test_input_unchanged(frame):
    before = frame.copy(deep=True)
    fingerprint_dataset(frame)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize('field,value', [
    ('microVolt', 30.125),
    ('ds', pd.Timestamp('2023-04-01')),
])
def test_changed_measurement_or_date_changes_hash(frame, field, value):
    modified = frame.copy(deep=True)
    modified.loc[0, field] = value
    assert fingerprint_dataset(modified).digest != fingerprint_dataset(frame).digest


def test_missing_inserted_row_changes_hash(frame):
    assert fingerprint_dataset(frame.iloc[[0, 2]]).digest != fingerprint_dataset(frame).digest


def test_observation_flag_cannot_be_changed_inconsistently(frame):
    frame.loc[1, 'is_observed'] = True
    with pytest.raises(FingerprintValidationError, match='observed value is missing'):
        fingerprint_dataset(frame)


def test_inserted_value_rejected(frame):
    frame.loc[1, 'microVolt'] = 30.2
    with pytest.raises(FingerprintValidationError, match='inserted period'):
        fingerprint_dataset(frame)


def test_duplicate_dates_rejected(frame):
    frame.loc[1, 'ds'] = frame.loc[0, 'ds']
    with pytest.raises(FingerprintValidationError, match='duplicate'):
        fingerprint_dataset(frame)


@pytest.mark.parametrize('bad', [float('inf'), float('-inf'), 'bad', True])
def test_invalid_target_rejected(frame, bad):
    frame['microVolt'] = frame['microVolt'].astype(object)
    frame.loc[0, 'microVolt'] = bad
    with pytest.raises(FingerprintValidationError):
        fingerprint_dataset(frame)


@pytest.mark.parametrize('bad', ['True', 1, None])
def test_inferred_or_invalid_flag_rejected(frame, bad):
    frame['is_observed'] = frame['is_observed'].astype(object)
    frame.loc[0, 'is_observed'] = bad
    with pytest.raises(FingerprintValidationError, match='explicit booleans'):
        fingerprint_dataset(frame)


def test_missing_columns_rejected(frame):
    with pytest.raises(FingerprintValidationError, match='missing required'):
        fingerprint_dataset(frame.drop(columns='is_observed'))


def test_empty_rejected(frame):
    with pytest.raises(FingerprintValidationError, match='empty'):
        fingerprint_dataset(frame.iloc[:0])


def test_timezone_aware_rejected(frame):
    frame['ds'] = frame['ds'].dt.tz_localize('UTC')
    with pytest.raises(FingerprintValidationError, match='timezone-naive'):
        fingerprint_dataset(frame)


def test_time_of_day_rejected(frame):
    frame.loc[0, 'ds'] = pd.Timestamp('2023-03-01 12:00')
    with pytest.raises(FingerprintValidationError, match='calendar date'):
        fingerprint_dataset(frame)


def test_missing_timestamp_rejected(frame):
    frame.loc[0, 'ds'] = pd.NaT
    with pytest.raises(FingerprintValidationError):
        fingerprint_dataset(frame)


def test_signed_zero_equivalent(frame):
    frame.loc[0, 'microVolt'] = 0.0
    first = fingerprint_dataset(frame)
    frame.loc[0, 'microVolt'] = -0.0
    assert fingerprint_dataset(frame) == first


def test_verification_and_mismatch(frame):
    record = fingerprint_dataset(frame)
    assert verify_dataset_fingerprint(frame, record)
    assert verify_dataset_fingerprint(frame, record.to_dict())
    frame.loc[0, 'microVolt'] += 0.01
    assert not verify_dataset_fingerprint(frame, record)


def test_wrong_row_count_returns_false(frame):
    record = fingerprint_dataset(frame).to_dict()
    record['row_count'] += 1
    assert not verify_dataset_fingerprint(frame, record)


@pytest.mark.parametrize('change', [
    {'canonicalization_version': 'v2'},
    {'algorithm': 'md5'},
    {'digest': 'not-a-digest'},
])
def test_invalid_verification_metadata_rejected(frame, change):
    expected = fingerprint_dataset(frame).to_dict()
    expected.update(change)
    with pytest.raises(FingerprintValidationError):
        verify_dataset_fingerprint(frame, expected)


def test_unversioned_manifest_subset_not_accepted_for_verification(frame):
    with pytest.raises(FingerprintValidationError, match='requires'):
        verify_dataset_fingerprint(frame, fingerprint_dataset(frame).to_manifest_fingerprint())


def test_file_hash_exact_bytes(tmp_path):
    path = tmp_path / 'data.csv'
    content = b'ds,microVolt,is_observed\r\n2023-03-01,30.0,True\r\n'
    path.write_bytes(content)
    assert fingerprint_file(path) == sha256(content).hexdigest()
    path.write_bytes(content.replace(b'\r\n', b'\n'))
    assert fingerprint_file(path) != sha256(content).hexdigest()


def test_csv_formatting_does_not_change_canonical_hash(tmp_path):
    a = tmp_path / 'a.csv'
    b = tmp_path / 'b.csv'
    a.write_text('ds,microVolt,is_observed\n2023-03-01,30.0,True\n2023-06-01,,False\n', encoding='utf-8')
    b.write_bytes(b'is_observed,microVolt,ds\r\nFalse,,2023-06-01\r\nTrue,30.000,2023-03-01\r\n')
    def read(path):
        data = pd.read_csv(path)
        data['ds'] = pd.to_datetime(data['ds'])
        return data
    assert fingerprint_file(a) != fingerprint_file(b)
    assert fingerprint_dataset(read(a)) == fingerprint_dataset(read(b))


def test_raw_battery_dataset_rejected(frame):
    from zenerestimation.data.dataset import BatteryDataset
    raw = BatteryDataset(frame)
    with pytest.raises(FingerprintValidationError, match='processed BatteryDataset'):
        fingerprint_dataset(raw)


def test_processed_battery_dataset_supported(frame):
    from zenerestimation.data.dataset import BatteryDataset
    processed = BatteryDataset(frame)
    processed._source_type = 'processed'
    assert fingerprint_dataset(processed) == fingerprint_dataset(frame)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fingerprint_file(tmp_path / 'missing.csv')
