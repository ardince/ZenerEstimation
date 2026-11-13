"""M18-C6-D3: deterministic, inspectable final forecast execution records.

Digests establish consistency against independently trusted planning inputs.
They do not authenticate authorization, model-code identity, or actual execution.
No model is instantiated, fitted, or predicted in this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from zenerestimation.forecasting.result import ForecastResult
from zenerestimation.optimization.internal_forecast_plan import verify_internal_forecast_plan


ARTIFACT_VERSION = "m18_c6_execution_artifact_v1"
ARTIFACT_TYPE = "authorized_forecast_execution"
EXECUTION_VERSION = "m18_c6_execution_v1"
_ARTIFACT_KEYS = frozenset({"artifact_version", "artifact_type", "execution_record", "artifact_digest"})
_RECORD_KEYS = frozenset({"execution_version", "status", "selected_model", "plan_digest", "decision_digest", "execution_evidence", "forecast"})
_EVIDENCE_KEYS = frozenset({"execution_authorized", "model_params_digest", "training_fingerprint", "forecast_horizon", "forecast_dates", "result_validated"})
_FORECAST_KEYS = frozenset({"model", "horizon", "dates", "values_microVolt"})


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + sha256(_canonical(value)).hexdigest()


def _plain_json(value: object) -> bool:
    """Disallow non-JSON values, non-string keys, NaN, and nonfinite numbers."""
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return isfinite(value)
    if type(value) is list:
        return all(_plain_json(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain_json(item) for key, item in value.items())
    return False


def _exact_dict(value: object, keys: frozenset[str]) -> bool:
    return type(value) is dict and set(value) == keys


def _is_digest(value: object) -> bool:
    return (type(value) is str and len(value) == 71 and
            value.startswith("sha256:") and
            all(c in "0123456789abcdef" for c in value[7:]))


def _plan_is_verified(plan: Mapping, *, artifact: Mapping, results: Mapping,
                      validation_context: Mapping, registry, training_dataset,
                      training_context: Mapping, forecast_horizon: int,
                      forecast_dates) -> bool:
    return verify_internal_forecast_plan(
        plan, artifact=artifact, results=results,
        validation_context=validation_context, registry=registry,
        training_dataset=training_dataset, training_context=training_context,
        forecast_horizon=forecast_horizon, forecast_dates=forecast_dates,
    ) is True


def _valid_record(record: object, plan: Mapping, training_context: Mapping) -> bool:
    if not _exact_dict(record, _RECORD_KEYS):
        return False
    if (record["execution_version"] != EXECUTION_VERSION or
            record["status"] != "EXECUTED" or
            type(record["selected_model"]) is not str or
            record["selected_model"] != plan["selected_model"] or
            record["plan_digest"] != plan["plan_digest"] or
            record["decision_digest"] != plan["decision_digest"]):
        return False
    evidence = record["execution_evidence"]
    if not _exact_dict(evidence, _EVIDENCE_KEYS):
        return False
    if (evidence["execution_authorized"] is not True or
            evidence["result_validated"] is not True or
            evidence["model_params_digest"] != plan["parameters_digest"] or
            evidence["training_fingerprint"] != training_context["dataset_fingerprint"] or
            type(evidence["forecast_horizon"]) is not int or
            evidence["forecast_horizon"] != plan["forecast_horizon"] or
            type(evidence["forecast_dates"]) is not list or
            evidence["forecast_dates"] != plan["forecast_dates"]):
        return False
    forecast = record["forecast"]
    if not _exact_dict(forecast, _FORECAST_KEYS):
        return False
    if (type(forecast["model"]) is not str or
            forecast["model"] != plan["selected_model"] or
            type(forecast["horizon"]) is not int or
            forecast["horizon"] != plan["forecast_horizon"] or
            type(forecast["dates"]) is not list or
            forecast["dates"] != plan["forecast_dates"]):
        return False
    values = forecast["values_microVolt"]
    return (type(values) is list and len(values) == plan["forecast_horizon"] and
            all(type(v) in (int, float) and isfinite(v) for v in values))


def _valid_artifact(value: object, plan: Mapping, training_context: Mapping) -> bool:
    if not _exact_dict(value, _ARTIFACT_KEYS) or not _plain_json(value):
        return False
    if (value["artifact_version"] != ARTIFACT_VERSION or
            value["artifact_type"] != ARTIFACT_TYPE or
            not _is_digest(value["artifact_digest"])):
        return False
    payload = {k: v for k, v in value.items() if k != "artifact_digest"}
    return (_digest(payload) == value["artifact_digest"] and
            _valid_record(value["execution_record"], plan, training_context))


def build_internal_forecast_execution_artifact(
    execution: Mapping, *, plan: Mapping, artifact: Mapping, results: Mapping,
    validation_context: Mapping, registry, training_dataset,
    training_context: Mapping, forecast_horizon: int, forecast_dates,
) -> dict:
    """Serialize a previously completed C6-C execution; never execute models."""
    if not _plan_is_verified(
        plan, artifact=artifact, results=results,
        validation_context=validation_context, registry=registry,
        training_dataset=training_dataset, training_context=training_context,
        forecast_horizon=forecast_horizon, forecast_dates=forecast_dates,
    ):
        raise ValueError("forecast plan does not match trusted inputs")
    execution_keys = _RECORD_KEYS - {"forecast"} | {"forecast_result"}
    if not _exact_dict(execution, execution_keys):
        raise ValueError("execution must have exact C6-C execution fields")
    result = execution["forecast_result"]
    if not isinstance(result, ForecastResult):
        raise TypeError("execution forecast_result must be ForecastResult")
    expected_dates = pd.DatetimeIndex(plan["forecast_dates"])
    if (type(result.model) is not str or result.model != plan["selected_model"] or
            type(result.horizon) is not int or result.horizon != plan["forecast_horizon"] or
            not isinstance(result.dates, pd.DatetimeIndex) or
            not result.dates.equals(expected_dates) or
            not isinstance(result.forecast, pd.Series) or
            len(result.forecast) != plan["forecast_horizon"] or
            not isinstance(result.forecast.index, pd.DatetimeIndex) or
            not result.forecast.index.equals(expected_dates)):
        raise ValueError("execution forecast result differs from approved plan")
    # Numeric strings and booleans are not legitimate forecast measurements.
    values = result.forecast.tolist()
    if not all(type(v) in (int, float, np.integer, np.floating) and
               isfinite(float(v)) for v in values):
        raise ValueError("forecast values must be finite numeric measurements")
    record = {
        "execution_version": execution["execution_version"],
        "status": execution["status"],
        "selected_model": execution["selected_model"],
        "plan_digest": execution["plan_digest"],
        "decision_digest": execution["decision_digest"],
        "execution_evidence": deepcopy(execution["execution_evidence"]),
        "forecast": {
            "model": result.model,
            "horizon": result.horizon,
            "dates": list(plan["forecast_dates"]),
            "values_microVolt": [float(v) for v in values],
        },
    }
    if not _plain_json(record) or not _valid_record(record, plan, training_context):
        raise ValueError("execution evidence is not consistent with trusted plan")
    payload = {"artifact_version": ARTIFACT_VERSION,
               "artifact_type": ARTIFACT_TYPE, "execution_record": record}
    output = {**payload, "artifact_digest": _digest(payload)}
    return deepcopy(output)


def verify_internal_forecast_execution_artifact(
    execution_artifact: Mapping, *, plan: Mapping, artifact: Mapping,
    results: Mapping, validation_context: Mapping, registry, training_dataset,
    training_context: Mapping, forecast_horizon: int, forecast_dates,
) -> bool:
    """Verify a persisted record against trusted plan inputs, without execution."""
    try:
        if not _plan_is_verified(
            plan, artifact=artifact, results=results,
            validation_context=validation_context, registry=registry,
            training_dataset=training_dataset, training_context=training_context,
            forecast_horizon=forecast_horizon, forecast_dates=forecast_dates,
        ):
            return False
        return _valid_artifact(execution_artifact, plan, training_context)
    except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
        return False


def save_internal_forecast_execution_artifact(execution_artifact: Mapping,
                                               path: str | Path) -> Path:
    """Atomically save a structurally valid, digest-consistent JSON artifact.

    Without trusted inputs this cannot revalidate selection or execution origin.
    Call verify_internal_forecast_execution_artifact separately before saving.
    """
    if not _exact_dict(execution_artifact, _ARTIFACT_KEYS) or not _plain_json(execution_artifact):
        raise ValueError("execution artifact is not canonical JSON data")
    if (execution_artifact["artifact_version"] != ARTIFACT_VERSION or
            execution_artifact["artifact_type"] != ARTIFACT_TYPE or
            not _is_digest(execution_artifact["artifact_digest"]) or
            not _exact_dict(execution_artifact["execution_record"], _RECORD_KEYS)):
        raise ValueError("execution artifact has an invalid schema")
    payload = {k: v for k, v in execution_artifact.items() if k != "artifact_digest"}
    if _digest(payload) != execution_artifact["artifact_digest"]:
        raise ValueError("execution artifact digest mismatch")
    # Persistence checks syntax and digest; trust-bound verification needs the
    # independent inputs that are intentionally absent from this API.
    target = Path(path)
    content = _canonical(execution_artifact)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temp_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
    return target
