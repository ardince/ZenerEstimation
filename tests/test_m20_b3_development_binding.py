"""M20-B3-C: real M18/M19 interface and adversarial binding tests."""
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
)
from zenerestimation.assessment.holdout_isolation import isolate_declared_holdout
from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.optimization.authorization import OptimizationExecutionRequest
from zenerestimation.optimization.development_binding import (
    build_validation_protocol_digest, verify_authorized_development_dataset,
)
from zenerestimation.optimization.internal_provenance import build_internal_validation_context
from zenerestimation.optimization.validation import ExpandingWindowSplitter


def scenario():
    dates = pd.date_range("2021-03-01", periods=14, freq="QS-MAR")
    frame = pd.DataFrame({
        "ds": dates,
        "microVolt": [float(i + 20) for i in range(14)],
        "is_observed": [True] * 14,
    })
    source = BatteryDataset(frame)
    declaration = AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id="m20-b3-410", assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id="410", development_cutoff="2023-12-01",
        forecast_dates=("2024-03-01", "2024-06-01"),
    )
    isolated = isolate_declared_holdout(source, declaration=declaration)
    training = deepcopy(isolated.development)
    splitter = ExpandingWindowSplitter(folds=2, validation_steps=2)
    context = build_internal_validation_context(training, splitter, dataset_id="410")
    request = OptimizationExecutionRequest(
        dataset_id="410", development_fingerprint=context["dataset_fingerprint"],
        development_cutoff="2023-12-01", model_family="LSTM", strategy="grid",
        max_trials=2, max_epoch_folds=100,
        validation_protocol_digest=build_validation_protocol_digest(context),
    )
    return request, declaration, source, training, splitter


def verify(items):
    request, declaration, source, training, splitter = items
    return verify_authorized_development_dataset(
        request, declaration=declaration, source_dataset=source,
        training_dataset=training, splitter=splitter,
    )


def test_valid_binding_detached_and_no_holdout_targets():
    items = scenario()
    result = verify(items)
    assert result.evidence["status"] == "DEVELOPMENT_BOUND"
    assert result.evidence["scientific_independence"] == "not_established"
    assert result.evidence["source_authenticity"] == "not_established"
    assert len(result.development.data) == 12
    assert "holdout" not in result.evidence
    assert "2024-03-01" not in result.development.data["ds"].dt.strftime("%Y-%m-%d").tolist()
    items[3].data.loc[0, "microVolt"] = 999
    assert result.development.data.loc[0, "microVolt"] == 20.0


def test_changed_training_measurement_rejected():
    items = list(scenario())
    items[3].data.loc[0, "microVolt"] = 999
    with pytest.raises(ValueError):
        verify(items)


def test_changed_source_measurement_rejected():
    items = list(scenario())
    items[2].data.loc[0, "microVolt"] = 999
    with pytest.raises(ValueError):
        verify(items)


def test_changed_training_timestamp_rejected():
    items = list(scenario())
    items[3].data.loc[0, "ds"] = pd.Timestamp("2020-12-01")
    with pytest.raises(ValueError):
        verify(items)


def test_changed_observed_mask_rejected():
    items = list(scenario())
    items[3].data.loc[0, "is_observed"] = False
    with pytest.raises(ValueError):
        verify(items)


def test_full_dataset_training_rejected():
    items = list(scenario())
    items[3] = items[2]
    with pytest.raises(ValueError):
        verify(items)


def test_wrong_dataset_id_rejected():
    items = list(scenario())
    items[0] = replace(items[0], dataset_id="110")
    with pytest.raises(ValueError, match="dataset_id"):
        verify(items)


def test_wrong_cutoff_rejected():
    items = list(scenario())
    items[0] = replace(items[0], development_cutoff="2023-09-01")
    with pytest.raises(ValueError, match="cutoff"):
        verify(items)


def test_wrong_protocol_rejected():
    items = list(scenario())
    items[4] = ExpandingWindowSplitter(folds=1, validation_steps=2)
    with pytest.raises(ValueError, match="protocol"):
        verify(items)


def test_wrong_authorized_fingerprint_rejected():
    items = list(scenario())
    items[0] = replace(items[0], development_fingerprint="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="fingerprint"):
        verify(items)


def test_source_holdout_change_does_not_affect_development_binding():
    items = list(scenario())
    items[2].data.loc[12:, "microVolt"] = [999.0, 998.0]
    result = verify(items)
    assert result.evidence["status"] == "DEVELOPMENT_BOUND"


def test_missing_observation_flags_fail_closed():
    items = list(scenario())
    items[2].data.drop(columns=["is_observed"], inplace=True)
    items[3].data.drop(columns=["is_observed"], inplace=True)
    with pytest.raises((ValueError, TypeError)):
        verify(items)


def test_protocol_digest_deterministic_and_sensitive():
    items = scenario()
    context = build_internal_validation_context(items[3], items[4], dataset_id="410")
    assert build_validation_protocol_digest(context) == build_validation_protocol_digest(context)
    modified = deepcopy(context)
    modified["validation_steps"] += 1
    assert build_validation_protocol_digest(modified) != build_validation_protocol_digest(context)
    with pytest.raises(ValueError):
        build_validation_protocol_digest({"protocol_version": "bad"})
