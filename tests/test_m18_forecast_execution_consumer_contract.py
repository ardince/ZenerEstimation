"""M18-C7-B1 RED contract: verified, read-only execution artifact consumption.

Install as tests/test_m18_forecast_execution_consumer_contract.py.
Do not create internal_forecast_execution_consumer.py until the RED gate is recorded.
The fixture executes a deterministic probe only during setup; consumer and loader
must never construct, fit, or predict. An unkeyed digest is not execution attestation.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

# Explicitly register transitive fixtures: pytest does not discover imported
# fixtures' dependencies merely because the dependent fixture is imported.
from tests.test_m18_forecast_plan_contract import training as training_fixture
from tests.test_m18_forecast_execution_contract import (
    ProbeModel, SubstituteModel, prepared as execution_prepared, execute,
)
from zenerestimation.optimization.internal_forecast_execution_artifact import (
    build_internal_forecast_execution_artifact,
    save_internal_forecast_execution_artifact,
)
from zenerestimation.optimization.internal_forecast_execution_consumer import (
    consume_internal_forecast_execution_artifact,
    load_internal_forecast_execution_artifact,
)

VIEW_VERSION = "m18_c7_forecast_view_v1"
VIEW_KEYS = {"view_version", "status", "selected_model", "forecast", "provenance"}
FORECAST_KEYS = {"horizon", "dates", "values_microVolt"}
PROVENANCE_KEYS = {
    "plan_digest", "decision_digest", "execution_artifact_digest", "training_fingerprint",
}


@pytest.fixture
def case(execution_prepared):
    plan, kwargs = execution_prepared
    execution = execute(plan, kwargs)
    artifact = build_internal_forecast_execution_artifact(execution, plan=plan, **kwargs)
    ProbeModel.events = []
    SubstituteModel.events = []
    return artifact, plan, kwargs


_UNSET = object()


def consume(case, artifact=_UNSET, **overrides):
    value, plan, kwargs = case
    trusted = dict(kwargs)
    trusted.update(overrides)
    return consume_internal_forecast_execution_artifact(
        value if artifact is _UNSET else artifact, plan=plan, **trusted,
    )


def load(case, path, **overrides):
    _, plan, kwargs = case
    trusted = dict(kwargs)
    trusted.update(overrides)
    return load_internal_forecast_execution_artifact(path, plan=plan, **trusted)


def assert_no_execution():
    assert ProbeModel.events == []
    assert SubstituteModel.events == []


def test_consume_exact_view_schema_and_values(case):
    value, plan, kwargs = case
    view = consume(case)
    assert type(view) is dict and set(view) == VIEW_KEYS
    assert view["view_version"] == VIEW_VERSION
    assert view["status"] == "VERIFIED"
    assert view["selected_model"] == "GRU"
    assert type(view["forecast"]) is dict and set(view["forecast"]) == FORECAST_KEYS
    assert view["forecast"] == {
        "horizon": 2,
        "dates": ["2024-03-01", "2024-06-01"],
        "values_microVolt": [29.4, 29.5],
    }
    assert type(view["provenance"]) is dict
    assert set(view["provenance"]) == PROVENANCE_KEYS
    assert view["provenance"] == {
        "plan_digest": plan["plan_digest"],
        "decision_digest": plan["decision_digest"],
        "execution_artifact_digest": value["artifact_digest"],
        "training_fingerprint": kwargs["training_context"]["dataset_fingerprint"],
    }
    assert_no_execution()


def test_consume_deterministic_json_safe(case):
    first = consume(case)
    second = consume(case)
    assert first == second
    encoded = json.dumps(first, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    assert json.loads(encoded) == first
    assert_no_execution()


def test_consume_detaches_output_and_does_not_mutate_inputs(case):
    value, plan, kwargs = case
    original_artifact = deepcopy(value)
    original_plan = deepcopy(plan)
    original_context = deepcopy(kwargs["training_context"])
    original_training = kwargs["training_dataset"].data.copy(deep=True)
    view = consume(case)
    view["forecast"]["values_microVolt"][0] = -900.0
    view["forecast"]["dates"][0] = "2000-01-01"
    view["provenance"]["training_fingerprint"]["digest"] = "0" * 64
    assert value == original_artifact
    assert plan == original_plan
    assert kwargs["training_context"] == original_context
    assert kwargs["training_dataset"].data.equals(original_training)
    assert consume(case)["forecast"]["values_microVolt"] == [29.4, 29.5]
    assert_no_execution()


def test_consume_no_filesystem_access(case, monkeypatch):
    # In-memory consumption must not use filesystem reads OR writes.
    def forbidden(*args, **kwargs):
        raise AssertionError("in-memory consumer accessed filesystem")
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    assert consume(case)["status"] == "VERIFIED"
    assert_no_execution()


@pytest.mark.parametrize("bad", [None, "GRU", 1, True, [], {}, {"status": "EXECUTED"}])
def test_consume_rejects_nonartifact_inputs(case, bad):
    with pytest.raises((TypeError, ValueError)):
        consume(case, artifact=bad)
    assert_no_execution()


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(artifact_version="unknown"),
    lambda x: x.update(artifact_type="selection"),
    lambda x: x.update(artifact_digest="sha256:" + "0" * 64),
    lambda x: x.update(extra=True),
    lambda x: x.pop("execution_record"),
    lambda x: x["execution_record"].update(selected_model="LSTM"),
    lambda x: x["execution_record"].update(status="PLANNED"),
    lambda x: x["execution_record"].update(plan_digest="sha256:" + "0" * 64),
    lambda x: x["execution_record"]["execution_evidence"].update(execution_authorized=False),
    lambda x: x["execution_record"]["execution_evidence"].update(result_validated=False),
    lambda x: x["execution_record"]["forecast"].update(dates=["2024-06-01", "2024-03-01"]),
    lambda x: x["execution_record"]["forecast"].update(horizon=3),
    lambda x: x["execution_record"]["forecast"].update(values_microVolt=[29.4]),
    lambda x: x["execution_record"]["forecast"].update(values_microVolt=[29.4, float("nan")]),
    lambda x: x["execution_record"]["forecast"].update(values_microVolt=[29.4, float("inf")]),
])
def test_consume_rejects_tampered_or_invalid_artifact(case, mutate):
    value = deepcopy(case[0])
    mutate(value)
    with pytest.raises((TypeError, ValueError)):
        consume(case, artifact=value)
    assert_no_execution()


@pytest.mark.parametrize("change", [
    lambda k: k["training_context"]["dataset_fingerprint"].update(digest="0" * 64),
    lambda k: k["validation_context"].update(dataset_id="other"),
    lambda k: k["artifact"]["selection_decision_record"]["decision"].update(selected_model="LSTM"),
])
def test_consume_rejects_mismatched_trusted_context(case, change):
    trusted = dict(case[2])
    trusted["training_context"] = deepcopy(trusted["training_context"])
    trusted["validation_context"] = deepcopy(trusted["validation_context"])
    trusted["artifact"] = deepcopy(trusted["artifact"])
    change(trusted)
    with pytest.raises((TypeError, ValueError)):
        consume_internal_forecast_execution_artifact(case[0], plan=case[1], **trusted)
    assert_no_execution()


def test_consume_rejects_mismatched_plan(case):
    plan = deepcopy(case[1])
    plan["plan_digest"] = "sha256:" + "0" * 64
    with pytest.raises((TypeError, ValueError)):
        consume_internal_forecast_execution_artifact(case[0], plan=plan, **case[2])
    assert_no_execution()


def test_consume_rejects_changed_training_dataset(case):
    kwargs = dict(case[2])
    kwargs["training_dataset"] = deepcopy(kwargs["training_dataset"])
    kwargs["training_dataset"].data.loc[0, "microVolt"] += 1.0
    with pytest.raises((TypeError, ValueError)):
        consume_internal_forecast_execution_artifact(case[0], plan=case[1], **kwargs)
    assert_no_execution()


def test_consume_rejects_missing_trusted_input(case):
    kwargs = dict(case[2])
    kwargs["registry"] = None
    with pytest.raises((TypeError, ValueError)):
        consume_internal_forecast_execution_artifact(case[0], plan=case[1], **kwargs)
    assert_no_execution()


def test_consume_does_not_rehash_or_repair_invalid_digest(case):
    altered = deepcopy(case[0])
    altered["artifact_digest"] = "sha256:" + "0" * 64
    original = deepcopy(altered)
    with pytest.raises((TypeError, ValueError)):
        consume(case, artifact=altered)
    assert altered == original
    assert_no_execution()


def test_load_valid_saved_artifact_roundtrip(case, tmp_path):
    target = tmp_path / "execution.json"
    save_internal_forecast_execution_artifact(case[0], target)
    before = target.read_bytes()
    listing = set(tmp_path.iterdir())
    view = load(case, target)
    assert view == consume(case)
    assert target.read_bytes() == before
    assert set(tmp_path.iterdir()) == listing
    assert_no_execution()


def test_load_accepts_path_string(case, tmp_path):
    target = tmp_path / "execution.json"
    save_internal_forecast_execution_artifact(case[0], target)
    assert load(case, str(target)) == consume(case)
    assert_no_execution()


def test_load_rejects_missing_file(case, tmp_path):
    with pytest.raises((OSError, ValueError)):
        load(case, tmp_path / "absent.json")
    assert_no_execution()


@pytest.mark.parametrize("payload", [
    "", "not json", "[]", "null", "{", "{\"artifact_digest\":\"x\"}",
    "{\"x\":NaN}", "{\"x\":Infinity}",
    '{"x":1,"x":2}',
])
def test_load_rejects_invalid_json_or_duplicate_keys(case, tmp_path, payload):
    target = tmp_path / "invalid.json"
    target.write_text(payload, encoding="utf-8")
    before = target.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        load(case, target)
    assert target.read_bytes() == before
    assert_no_execution()


def test_load_rejects_duplicate_keys_in_otherwise_valid_artifact(case, tmp_path):
    # Duplicate root key must be rejected, even if both values are identical.
    raw = json.dumps(case[0], sort_keys=True, separators=(",", ":"))
    assert raw.startswith("{")
    target = tmp_path / "duplicate.json"
    target.write_text('{"artifact_type":"authorized_forecast_execution",' + raw[1:], encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load(case, target)
    assert_no_execution()


def test_load_rejects_tampered_saved_artifact(case, tmp_path):
    target = tmp_path / "tampered.json"
    altered = deepcopy(case[0])
    altered["execution_record"]["forecast"]["values_microVolt"][0] = -1.0
    target.write_text(json.dumps(altered), encoding="utf-8")
    before = target.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        load(case, target)
    assert target.read_bytes() == before
    assert_no_execution()


def test_load_rejects_oversized_file(case, tmp_path):
    # C7-A requires a finite maximum JSON size, but does not yet fix its
    # numerical value. 16 MiB is deliberately above a sensible evidence cap.
    target = tmp_path / "oversized.json"
    target.write_bytes(b" " * (16 * 1024 * 1024 + 1))
    with pytest.raises((OSError, ValueError)):
        load(case, target)
    assert_no_execution()


def test_load_does_not_reexecute_or_change_training(case, tmp_path):
    target = tmp_path / "execution.json"
    save_internal_forecast_execution_artifact(case[0], target)
    original = case[2]["training_dataset"].data.copy(deep=True)
    load(case, target)
    assert case[2]["training_dataset"].data.equals(original)
    assert_no_execution()
