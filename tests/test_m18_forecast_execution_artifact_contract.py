"""M18-C6-D2 RED contract: durable, independently verifiable execution evidence.

Install at tests/test_m18_forecast_execution_artifact_contract.py.
Do NOT create internal_forecast_execution_artifact.py before the RED run.

Trust boundary: SHA-256 checks consistency, NOT human authorization,
model-code identity, or proof that a recorded model actually ran.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from tests.test_m18_forecast_execution_contract import (
    ProbeModel,
    prepared as execution_prepared,
    execute,
)

from tests.test_m18_forecast_plan_contract import (
    training as training_fixture,
)

from zenerestimation.optimization.internal_forecast_execution_artifact import (
    build_internal_forecast_execution_artifact,
    verify_internal_forecast_execution_artifact,
    save_internal_forecast_execution_artifact,
)

VERSION = "m18_c6_execution_artifact_v1"
ARTIFACT_KEYS = {"artifact_version", "artifact_type", "execution_record", "artifact_digest"}
RECORD_KEYS = {
    "execution_version", "status", "selected_model", "plan_digest",
    "decision_digest", "execution_evidence", "forecast",
}
EVIDENCE_KEYS = {
    "execution_authorized", "model_params_digest", "training_fingerprint",
    "forecast_horizon", "forecast_dates", "result_validated",
}
FORECAST_KEYS = {"model", "horizon", "dates", "values_microVolt"}


def digest(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return "sha256:" + sha256(raw).hexdigest()


def reseal(value):
    """Simulate an actor able to rewrite content AND recompute an unkeyed hash."""
    value["artifact_digest"] = digest({k: v for k, v in value.items() if k != "artifact_digest"})
    return value


@pytest.fixture
def case(execution_prepared):
    plan, kwargs = execution_prepared
    execution = execute(plan, kwargs)
    # Reset the probe so verification can prove it never executes a model.
    ProbeModel.events = []
    return execution, plan, kwargs


def build(case):
    execution, plan, kwargs = case
    return build_internal_forecast_execution_artifact(execution, plan=plan, **kwargs)


def verify(value, case):
    _, plan, kwargs = case
    return verify_internal_forecast_execution_artifact(value, plan=plan, **kwargs)


def test_build_exact_schema_and_forecast_values(case):
    value = build(case)
    execution, plan, kwargs = case
    assert type(value) is dict and set(value) == ARTIFACT_KEYS
    assert value["artifact_version"] == VERSION
    assert value["artifact_type"] == "authorized_forecast_execution"
    record = value["execution_record"]
    assert type(record) is dict and set(record) == RECORD_KEYS
    assert record["execution_version"] == "m18_c6_execution_v1"
    assert record["status"] == "EXECUTED"
    assert record["selected_model"] == plan["selected_model"] == "GRU"
    assert record["plan_digest"] == plan["plan_digest"]
    assert record["decision_digest"] == plan["decision_digest"]
    assert type(record["forecast"]) is dict and set(record["forecast"]) == FORECAST_KEYS
    assert record["forecast"] == {
        "model": "GRU", "horizon": 2,
        "dates": ["2024-03-01", "2024-06-01"],
        "values_microVolt": [29.4, 29.5],
    }
    evidence = record["execution_evidence"]
    assert type(evidence) is dict and set(evidence) == EVIDENCE_KEYS
    assert evidence["execution_authorized"] is True
    assert evidence["result_validated"] is True
    assert evidence["model_params_digest"] == plan["parameters_digest"]
    assert evidence["training_fingerprint"] == kwargs["training_context"]["dataset_fingerprint"]
    assert evidence["forecast_horizon"] == 2
    assert evidence["forecast_dates"] == plan["forecast_dates"]
    assert value["artifact_digest"] == digest({k: v for k, v in value.items() if k != "artifact_digest"})
    assert verify(value, case) is True


def test_json_safe_and_deterministic(case):
    first, second = build(case), build(case)
    assert first == second
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert json.dumps(first, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8") == json.dumps(
        second, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")


def test_builder_does_not_mutate_execution_or_trusted_inputs(case):
    execution, plan, kwargs = case
    original_evidence = deepcopy(execution["execution_evidence"])
    original_values = execution["forecast_result"].forecast.copy(deep=True)
    original_plan = deepcopy(plan)
    original_training = kwargs["training_dataset"].data.copy(deep=True)
    original_context = deepcopy(kwargs["training_context"])
    value = build(case)
    value["execution_record"]["forecast"]["values_microVolt"][0] = -999.0
    value["execution_record"]["execution_evidence"]["training_fingerprint"]["digest"] = "0" * 64
    assert execution["execution_evidence"] == original_evidence
    assert execution["forecast_result"].forecast.equals(original_values)
    assert plan == original_plan
    assert kwargs["training_dataset"].data.equals(original_training)
    assert kwargs["training_context"] == original_context


def test_verify_does_not_construct_fit_or_predict(case):
    value = build(case)
    assert verify(value, case) is True
    assert ProbeModel.events == []


def test_verify_does_not_modify_artifact_or_trusted_inputs(case):
    value = build(case)
    _, plan, kwargs = case
    before_value = deepcopy(value)
    before_plan = deepcopy(plan)
    before_context = deepcopy(kwargs["training_context"])
    before_data = kwargs["training_dataset"].data.copy(deep=True)
    assert verify(value, case) is True
    assert value == before_value and plan == before_plan
    assert kwargs["training_context"] == before_context
    assert kwargs["training_dataset"].data.equals(before_data)


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(artifact_version="m18_c6_execution_artifact_v2"),
    lambda x: x.update(artifact_type="selection_artifact"),
    lambda x: x.update(extra=True),
    lambda x: x.pop("execution_record"),
    lambda x: x.update(artifact_digest="sha256:" + "0" * 64),
    lambda x: x["execution_record"].update(status="PLANNED"),
    lambda x: x["execution_record"].update(selected_model="LSTM"),
    lambda x: x["execution_record"].update(plan_digest="sha256:" + "0" * 64),
    lambda x: x["execution_record"].update(decision_digest="sha256:" + "0" * 64),
    lambda x: x["execution_record"]["execution_evidence"].update(execution_authorized=False),
    lambda x: x["execution_record"]["execution_evidence"].update(result_validated=False),
    lambda x: x["execution_record"]["execution_evidence"].update(model_params_digest="sha256:" + "0" * 64),
    lambda x: x["execution_record"]["execution_evidence"]["training_fingerprint"].update(digest="0" * 64),
    lambda x: x["execution_record"]["execution_evidence"].update(forecast_horizon=3),
    lambda x: x["execution_record"]["execution_evidence"]["forecast_dates"].reverse(),
    lambda x: x["execution_record"]["forecast"].update(model="LSTM"),
    lambda x: x["execution_record"]["forecast"].update(horizon=3),
    lambda x: x["execution_record"]["forecast"]["dates"].reverse(),
    lambda x: x["execution_record"]["forecast"]["values_microVolt"].__setitem__(0, 99.0),
    lambda x: x["execution_record"]["forecast"]["values_microVolt"].pop(),
    lambda x: x["execution_record"]["forecast"].update(extra="unexpected"),
])
def test_unresealed_tampering_is_rejected(case, mutate):
    value = build(case)
    mutate(value)
    assert verify(value, case) is False


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(artifact_version="unsupported"),
    lambda x: x.update(artifact_type="other"),
    lambda x: x.update(extra=123),
    lambda x: x["execution_record"].update(extra=123),
    lambda x: x["execution_record"].update(status="PLANNED"),
    lambda x: x["execution_record"].update(selected_model="LSTM"),
    lambda x: x["execution_record"].update(plan_digest="sha256:" + "0" * 64),
    lambda x: x["execution_record"].update(decision_digest="sha256:" + "0" * 64),
    lambda x: x["execution_record"]["execution_evidence"].update(extra=True),
    lambda x: x["execution_record"]["execution_evidence"].update(execution_authorized=1),
    lambda x: x["execution_record"]["execution_evidence"].update(result_validated=1),
    lambda x: x["execution_record"]["execution_evidence"].update(model_params_digest="sha256:" + "0" * 64),
    lambda x: x["execution_record"]["execution_evidence"]["training_fingerprint"].update(digest="0" * 64),
    lambda x: x["execution_record"]["execution_evidence"].update(forecast_horizon=True),
    lambda x: x["execution_record"]["execution_evidence"].update(forecast_dates=["2024-06-01", "2024-03-01"]),
    lambda x: x["execution_record"]["forecast"].update(extra=True),
    lambda x: x["execution_record"]["forecast"].update(model="LSTM"),
    lambda x: x["execution_record"]["forecast"].update(horizon=True),
    lambda x: x["execution_record"]["forecast"].update(dates=["2024-06-01", "2024-03-01"]),
    lambda x: x["execution_record"]["forecast"].update(values_microVolt=[29.4]),
    lambda x: x["execution_record"]["forecast"].update(values_microVolt=[29.4, True]),
    lambda x: x["execution_record"]["forecast"].update(values_microVolt=[29.4, "29.5"]),
    lambda x: x["execution_record"]["forecast"].update(values_microVolt=[29.4, float("nan")]),
    lambda x: x["execution_record"]["forecast"].update(values_microVolt=[29.4, float("inf")]),
])
def test_resealed_structural_or_provenance_forgery_rejected(case, mutate):
    value = build(case)
    mutate(value)
    try:
        reseal(value)
    except (TypeError, ValueError):
        # Nonfinite JSON is invalid before hashing; verifier must reject it too.
        pass
    assert verify(value, case) is False


def test_rehashed_forecast_value_is_not_authenticated_execution_proof(case):
    """An unkeyed digest cannot prove forecast origin or prevent deliberate rehashing.

    Without a trusted signed/attested execution record, a structurally valid
    altered value may remain internally consistent. This is NOT a model rerun.
    """
    value = build(case)
    value["execution_record"]["forecast"]["values_microVolt"][0] = 29.45
    reseal(value)
    assert verify(value, case) is True
    assert ProbeModel.events == []


@pytest.mark.parametrize("change", [
    lambda c: c[2]["training_dataset"].data.loc.__setitem__((0, "microVolt"), 99.0),
    lambda c: c[2]["training_context"]["dataset_fingerprint"].update(digest="0" * 64),
    lambda c: c[2]["artifact"]["selection_decision_record"]["decision"].update(selected_model="LSTM"),
    lambda c: c[1].update(plan_digest="sha256:" + "0" * 64),
    lambda c: c[2]["registry"]._models.clear(),
])
def test_changed_trusted_inputs_rejected(case, change):
    value = build(case)
    change(case)
    assert verify(value, case) is False
    assert ProbeModel.events == []


def invalidate_forecast_values(execution):
    """Inject a non-finite forecast value for negative contract testing."""
    forecast = execution["forecast_result"].forecast
    forecast.loc[forecast.index[0]] = float("nan")

@pytest.mark.parametrize("mutate", [
    lambda e: e.update(status="PLANNED"),
    lambda e: e.update(selected_model="LSTM"),
    lambda e: e.update(plan_digest="sha256:" + "0" * 64),
    lambda e: e.update(decision_digest="sha256:" + "0" * 64),
    lambda e: e.update(extra="unexpected"),
    lambda e: e["execution_evidence"].update(execution_authorized=False),
    lambda e: e["execution_evidence"].update(result_validated=False),
    lambda e: e["execution_evidence"].update(model_params_digest="sha256:" + "0" * 64),
    lambda e: e["execution_evidence"].update(forecast_dates=["2024-06-01", "2024-03-01"]),
    invalidate_forecast_values,
])

def test_builder_rejects_invalid_execution_input(case, mutate):
    execution, _, _ = case
    mutate(execution)
    with pytest.raises((TypeError, ValueError)):
        build(case)
    assert ProbeModel.events == []


def test_builder_rejects_missing_or_nonmapping_execution(case):
    for invalid in (None, [], "EXECUTED", {}):
        with pytest.raises((TypeError, ValueError)):
            build_internal_forecast_execution_artifact(invalid, plan=case[1], **case[2])


def test_build_and_verify_do_not_write_files(case, monkeypatch):
    import builtins
    def deny_write(*args, **kwargs):
        raise AssertionError("builder/verifier must not open files")
    monkeypatch.setattr(builtins, "open", deny_write)
    value = build(case)
    assert verify(value, case) is True


def test_atomic_save_and_utf8_roundtrip(case, tmp_path):
    value = build(case)
    target = tmp_path / "evidence.json"
    existing = set(tmp_path.iterdir())
    returned = save_internal_forecast_execution_artifact(value, target)
    assert Path(returned) == target
    assert json.loads(target.read_text(encoding="utf-8")) == value
    assert verify(json.loads(target.read_text(encoding="utf-8")), case) is True
    assert set(tmp_path.iterdir()) == existing | {target}


def test_save_deterministic_bytes_and_replace(case, tmp_path):
    value = build(case)
    target = tmp_path / "execution.json"
    existing = set(tmp_path.iterdir())
    save_internal_forecast_execution_artifact(value, target)
    first = target.read_bytes()
    target.write_text("old content", encoding="utf-8")
    save_internal_forecast_execution_artifact(value, target)
    assert target.read_bytes() == first
    assert set(tmp_path.iterdir()) == existing | {target}


def test_save_rejects_bad_digest_without_overwriting(case, tmp_path):
    value = build(case)
    target = tmp_path / "execution.json"
    target.write_bytes(b"existing valid file")
    existing = set(tmp_path.iterdir())
    value["execution_record"]["forecast"]["values_microVolt"][0] = -999.0
    with pytest.raises((TypeError, ValueError)):
        save_internal_forecast_execution_artifact(value, target)
    assert target.read_bytes() == b"existing valid file"
    assert set(tmp_path.iterdir()) == existing | {target}


def test_save_rejects_non_json_value_without_writing(case, tmp_path):
    value = build(case)
    value["execution_record"]["forecast"]["values_microVolt"][0] = float("nan")
    target = tmp_path / "invalid.json"
    with pytest.raises((TypeError, ValueError)):
        save_internal_forecast_execution_artifact(value, target)
    assert not target.exists()


def test_save_does_not_mutate_artifact(case, tmp_path):
    value = build(case)
    original = deepcopy(value)
    save_internal_forecast_execution_artifact(value, tmp_path / "execution.json")
    assert value == original
