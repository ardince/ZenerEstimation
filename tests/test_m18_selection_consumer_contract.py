"""M18-C4-B: RED-phase contract for read-only verified selection consumption.

Install in tests/. The deliberately absent production module is
zenerestimation.optimization.internal_selection_consumer.

Fixtures reuse the existing real C2 internal-validation test data.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from tests.test_m18_selection_audit_contract import context, result, unique_results
from zenerestimation.optimization.internal_selection_artifact import (
    build_internal_selection_artifact,
)
from zenerestimation.optimization.internal_selection_consumer import (
    consume_internal_selection_artifact,
)

VERSION = "m18_c4_consumption_v1"
FIELDS = {
    "consumption_version", "status", "selected_model",
    "decision_digest", "artifact_version",
}


def artifact(*, results=None, trusted=None, authorized=True):
    return build_internal_selection_artifact(
        unique_results() if results is None else results,
        validation_context=context() if trusted is None else trusted,
        authorized=authorized,
    )


def consume(value, *, results=None, trusted=None):
    return consume_internal_selection_artifact(
        value,
        results=unique_results() if results is None else results,
        validation_context=context() if trusted is None else trusted,
    )


def resign(value):
    record = value["selection_decision_record"]
    payload = {key: item for key, item in record.items() if key != "decision_digest"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    record["decision_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return value


def tied_results():
    return {
        "LSTM": result("LSTM", rmse=(0.2, 0.4)),
        "GRU": result("GRU", rmse=(0.2, 0.4)),
    }


def test_authorized_unique_winner_returns_exact_ready_contract():
    value = artifact(authorized=True)
    output = consume(value)
    assert type(output) is dict
    assert set(output) == FIELDS
    assert output == {
        "consumption_version": VERSION,
        "status": "READY",
        "selected_model": "GRU",
        "decision_digest": value["selection_decision_record"]["decision_digest"],
        "artifact_version": "m18_c3_selection_artifact_v1",
    }


def test_ready_output_is_deterministic():
    value = artifact()
    assert consume(value) == consume(value)
    assert json.dumps(consume(value), sort_keys=True, allow_nan=False)


def test_ready_model_is_exactly_verified_selected_model():
    results = unique_results()
    value = artifact(results=results)
    assert consume(value, results=results)["selected_model"] == (
        value["selection_decision_record"]["decision"]["selected_model"]
    )


def test_consumption_does_not_mutate_inputs():
    results, trusted = unique_results(), context()
    value = artifact(results=results, trusted=trusted)
    before_value = deepcopy(value)
    before_context = deepcopy(trusted)
    before_results = {name: item.to_dict() for name, item in results.items()}
    consume(value, results=results, trusted=trusted)
    assert value == before_value
    assert trusted == before_context
    assert {name: item.to_dict() for name, item in results.items()} == before_results


def test_output_is_detached_from_artifact():
    value = artifact()
    output = consume(value)
    output["selected_model"] = "ARIMA"
    assert value["selection_decision_record"]["decision"]["selected_model"] == "GRU"
    assert consume(value)["selected_model"] == "GRU"


@pytest.mark.parametrize("authorized", [False, True])
def test_unresolved_tie_is_rejected(authorized):
    results = tied_results()
    value = artifact(results=results, authorized=authorized)
    assert value["selection_decision_record"]["decision"]["status"] == "TIE_UNRESOLVED"
    with pytest.raises(ValueError):
        consume(value, results=results)


def test_pending_authorization_is_rejected():
    value = artifact(authorized=False)
    assert value["selection_decision_record"]["decision"]["status"] == "PENDING_AUTHORIZATION"
    with pytest.raises(ValueError):
        consume(value)


@pytest.mark.parametrize("field, replacement", [
    ("status", "PENDING_AUTHORIZATION"),
    ("status", "TIE_UNRESOLVED"),
    ("status", "ACTIVATED"),
    ("authorized", False),
    ("authorized", 1),
    ("selected_model", None),
    ("selected_model", "LSTM"),
    ("recommended_model", "LSTM"),
    ("selection_score", -1.0),
])
def test_resigned_modified_decision_is_rejected(field, replacement):
    value = artifact()
    value["selection_decision_record"]["decision"][field] = replacement
    resign(value)
    with pytest.raises(ValueError):
        consume(value)


@pytest.mark.parametrize("change", [
    lambda value: value.update(artifact_version="other"),
    lambda value: value.update(artifact_type="benchmark"),
    lambda value: value.update(extra="unexpected"),
    lambda value: value.pop("selection_decision_record"),
    lambda value: value["selection_decision_record"].update(decision_digest="sha256:" + "0" * 64),
    lambda value: value["selection_decision_record"]["decision"].update(selected_model="LSTM"),
])
def test_invalid_or_tampered_artifact_is_rejected(change):
    value = artifact()
    change(value)
    with pytest.raises(ValueError):
        consume(value)


def test_resigned_provenance_change_is_rejected():
    value = artifact()
    value["selection_decision_record"]["validation_provenance"]["dataset_id"] = "other"
    resign(value)
    with pytest.raises(ValueError):
        consume(value)


def test_mismatched_independent_context_is_rejected():
    value = artifact()
    with pytest.raises(ValueError):
        consume(value, trusted=context(dataset_id="other"))


def test_mismatched_independent_results_are_rejected():
    value = artifact()
    changed = unique_results()
    changed["GRU"] = result("GRU", rmse=(0.7, 0.8))
    with pytest.raises(ValueError):
        consume(value, results=changed)


def test_no_model_factory_or_forecast_execution(monkeypatch):
    # Consumption should use verified evidence only, not instantiate models.
    from zenerestimation.optimization import internal_selection_consumer as module
    assert not hasattr(module, "ForecastPipeline")
    assert not hasattr(module, "GenericOptimizer")
    assert not hasattr(module, "model_factory")
    assert consume(artifact())["status"] == "READY"


def test_no_filesystem_writes(tmp_path, monkeypatch):
    # A read-only consumer must not create files, including in the cwd.
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    assert consume(artifact())["status"] == "READY"
    assert set(tmp_path.iterdir()) == before


def test_rejects_raw_decision_record_without_c3_wrapper():
    value = artifact()
    with pytest.raises((TypeError, ValueError)):
        consume(value["selection_decision_record"])


def test_rejects_model_name_instead_of_artifact():
    with pytest.raises((TypeError, ValueError)):
        consume("GRU")


def test_rejects_missing_trusted_inputs():
    value = artifact()
    with pytest.raises((TypeError, ValueError)):
        consume_internal_selection_artifact(value, results=None, validation_context=context())
    with pytest.raises((TypeError, ValueError)):
        consume_internal_selection_artifact(value, results=unique_results(), validation_context=None)
