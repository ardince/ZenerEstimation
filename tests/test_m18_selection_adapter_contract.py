"""M18-C5-B RED contract: verified, side-effect-free model resolution.

Install at tests/test_m18_selection_adapter_contract.py.
The production module internal_selection_adapter intentionally does not exist yet.
"""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tests.test_m18_selection_audit_contract import context, result, unique_results
from zenerestimation.forecasting.base import BaseForecastModel
from zenerestimation.forecasting.models import ModelRegistry
from zenerestimation.optimization.internal_selection_artifact import (
    build_internal_selection_artifact,
)
from zenerestimation.optimization.internal_selection_adapter import (
    prepare_selected_forecast,
)

VERSION = "m18_c5_preparation_v1"
FIELDS = {
    "preparation_version", "status", "selected_model", "registry_model",
    "decision_digest", "artifact_version", "consumption_version",
}


class NeverExecuteModel(BaseForecastModel):
    """Registration is legal; any instantiation or execution is a failure."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("C5 must not instantiate a forecasting model")

    def fit(self, dataset):
        raise AssertionError("C5 must not fit a forecasting model")

    def predict(self, steps=1):
        raise AssertionError("C5 must not predict")


class AlternativeModel(BaseForecastModel):
    def fit(self, dataset):
        raise AssertionError("C5 must not fit")

    def predict(self, steps=1):
        raise AssertionError("C5 must not predict")


def registry(*, include_gru=True, include_lstm=False, name="GRU"):
    value = ModelRegistry()
    if include_gru:
        value.register(name, NeverExecuteModel)
    if include_lstm:
        value.register("LSTM", AlternativeModel)
    return value


def artifact(*, results=None, trusted=None, authorized=True):
    return build_internal_selection_artifact(
        unique_results() if results is None else results,
        validation_context=context() if trusted is None else trusted,
        authorized=authorized,
    )


def prepare(value, *, models=None, results=None, trusted=None):
    return prepare_selected_forecast(
        value,
        results=unique_results() if results is None else results,
        validation_context=context() if trusted is None else trusted,
        registry=registry() if models is None else models,
    )


def tied_results():
    return {
        "LSTM": result("LSTM", rmse=(0.2, 0.4)),
        "GRU": result("GRU", rmse=(0.2, 0.4)),
    }


def test_authorized_selection_produces_exact_prepared_contract():
    value = artifact()
    output = prepare(value)
    assert type(output) is dict
    assert set(output) == FIELDS
    assert output == {
        "preparation_version": VERSION,
        "status": "PREPARED",
        "selected_model": "GRU",
        "registry_model": "gru",
        "decision_digest": value["selection_decision_record"]["decision_digest"],
        "artifact_version": "m18_c3_selection_artifact_v1",
        "consumption_version": "m18_c4_consumption_v1",
    }


def test_preparation_is_deterministic_and_json_compatible():
    value, models = artifact(), registry()
    first = prepare(value, models=models)
    second = prepare(value, models=models)
    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True, allow_nan=False)) == first


@pytest.mark.parametrize("registered_name", ["GRU", "gru", "GrU"])
def test_registry_case_normalization_preserves_selected_identity(registered_name):
    output = prepare(artifact(), models=registry(name=registered_name))
    assert output["selected_model"] == "GRU"
    assert output["registry_model"] == "gru"


def test_registry_may_contain_other_models_without_substitution():
    output = prepare(artifact(), models=registry(include_lstm=True))
    assert output["selected_model"] == "GRU"


def test_unknown_selected_model_fails_even_if_other_models_exist():
    with pytest.raises(ValueError):
        prepare(artifact(), models=registry(include_gru=False, include_lstm=True))


def test_empty_registry_fails_closed():
    with pytest.raises(ValueError):
        prepare(artifact(), models=registry(include_gru=False))


@pytest.mark.parametrize("bad_registry", [None, {}, object(), "GRU", 123])
def test_registry_must_be_explicit_valid_model_registry(bad_registry):
    with pytest.raises((TypeError, ValueError)):
        prepare_selected_forecast(
            artifact(), results=unique_results(),
            validation_context=context(), registry=bad_registry,
        )


@pytest.mark.parametrize("authorized", [False, True])
def test_tie_is_never_prepared(authorized):
    results = tied_results()
    with pytest.raises((TypeError, ValueError)):
        prepare(artifact(results=results, authorized=authorized), results=results)


def test_pending_authorization_is_not_prepared():
    with pytest.raises((TypeError, ValueError)):
        prepare(artifact(authorized=False))


@pytest.mark.parametrize("change", [
    lambda value: value.update(artifact_version="wrong"),
    lambda value: value.update(artifact_type="wrong"),
    lambda value: value["selection_decision_record"].update(decision_digest="sha256:" + "0" * 64),
    lambda value: value["selection_decision_record"]["decision"].update(selected_model="LSTM"),
])
def test_tampered_artifact_fails_closed(change):
    value = artifact()
    change(value)
    with pytest.raises((TypeError, ValueError)):
        prepare(value)


def test_changed_trusted_results_fail_closed():
    value = artifact()
    changed = {
        "LSTM": result("LSTM", rmse=(0.1, 0.1)),
        "GRU": result("GRU", rmse=(0.2, 0.4)),
    }
    with pytest.raises((TypeError, ValueError)):
        prepare(value, results=changed)


def test_changed_trusted_validation_context_fails_closed():
    value = artifact()
    with pytest.raises((TypeError, ValueError)):
        prepare(value, trusted=context(dataset_id="different-dataset"))


def test_c4_ready_record_is_not_a_substitute_for_artifact():
    from zenerestimation.optimization.internal_selection_consumer import (
        consume_internal_selection_artifact,
    )
    value = artifact()
    ready = consume_internal_selection_artifact(
        value, results=unique_results(), validation_context=context(),
    )
    with pytest.raises((TypeError, ValueError)):
        prepare(ready)


def test_bare_selected_model_name_is_not_sufficient():
    with pytest.raises((TypeError, ValueError)):
        prepare("GRU")


def test_preparation_does_not_mutate_artifact_context_results_or_registry():
    results, trusted, models = unique_results(), context(), registry(include_lstm=True)
    value = artifact(results=results, trusted=trusted)
    original_artifact, original_context = deepcopy(value), deepcopy(trusted)
    original_results = {name: candidate.to_dict() for name, candidate in results.items()}
    original_models = dict(models._models)
    prepare(value, models=models, results=results, trusted=trusted)
    assert value == original_artifact
    assert trusted == original_context
    assert {name: candidate.to_dict() for name, candidate in results.items()} == original_results
    assert models._models == original_models


def test_preparation_record_is_detached_from_inputs():
    value, models = artifact(), registry()
    output = prepare(value, models=models)
    output["selected_model"] = "LSTM"
    output["status"] = "EXECUTED"
    assert prepare(value, models=models)["selected_model"] == "GRU"
    assert prepare(value, models=models)["status"] == "PREPARED"


def test_preparation_does_not_construct_fit_or_predict():
    # NeverExecuteModel.__init__ raises if invoked, and registry lookup alone
    # cannot invoke its fit/predict methods.
    assert prepare(artifact(), models=registry())["status"] == "PREPARED"


def test_preparation_does_not_call_forecast_pipeline(monkeypatch):
    from zenerestimation.forecasting.pipeline import ForecastPipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("C5 must not call ForecastPipeline.run")

    monkeypatch.setattr(ForecastPipeline, "run", forbidden)
    assert prepare(artifact())["status"] == "PREPARED"


def test_preparation_does_not_write_files(monkeypatch):
    import builtins
    original_open = builtins.open

    def no_write(file, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            raise AssertionError("C5 must not write files")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", no_write)
    assert prepare(artifact())["status"] == "PREPARED"


def test_preparation_record_does_not_claim_execution_or_verified_human_identity():
    output = prepare(artifact())
    assert "forecast" not in output
    assert "model_instance" not in output
    assert "model_parameters" not in output
    assert "human_identity_verified" not in output
    assert output["status"] == "PREPARED"


def test_missing_trusted_inputs_fail_closed():
    value = artifact()
    for results, trusted in ((None, context()), (unique_results(), None)):
        with pytest.raises((TypeError, ValueError)):
            prepare_selected_forecast(
                value, results=results, validation_context=trusted,
                registry=registry(),
            )
