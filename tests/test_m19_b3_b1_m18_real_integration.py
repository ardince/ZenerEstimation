"""M19-B3-B1 real M18 execution and A1 binding integration.

No M18 verifier is mocked. Uses the deterministic ProbeModel fixture; this
is not evidence of historical non-exposure or universal production enforcement.
"""
from copy import deepcopy

import pandas as pd
import pytest

from tests.test_m18_forecast_plan_contract import training as processed_training, inputs
from tests.test_m18_forecast_execution_contract import ProbeModel
from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
)
from zenerestimation.assessment.holdout_binding import verify_holdout_binding
from zenerestimation.assessment.holdout_evaluation_eligibility import (
    establish_holdout_evaluation_eligibility,
)
from zenerestimation.assessment.holdout_isolation import (
    isolate_declared_holdout, require_isolated_training_dataset,
)
from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.fingerprint import fingerprint_dataset
from zenerestimation.forecasting.models import ModelRegistry
from zenerestimation.optimization.internal_forecast_plan import (
    build_internal_forecast_plan, verify_internal_forecast_plan,
)
from zenerestimation.optimization.internal_forecast_execution import (
    execute_internal_forecast_plan,
)
from zenerestimation.optimization.internal_forecast_execution_artifact import (
    build_internal_forecast_execution_artifact,
    verify_internal_forecast_execution_artifact,
)


@pytest.fixture
def declaration():
    return AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id="m19-b3-b1-m18-real-path",
        assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id="410",
        development_cutoff="2023-12-01",
        forecast_dates=("2024-03-01", "2024-06-01"),
    )


@pytest.fixture
def full_frame(processed_training):
    development = processed_training.data[["ds", "microVolt"]].copy(deep=True)
    future = pd.DataFrame({
        "ds": pd.to_datetime(["2024-03-01", "2024-06-01"]),
        "microVolt": [29.9, 30.506],
    })
    return pd.concat([development, future], ignore_index=True)


def _real_case(processed_training, full_frame, declaration):
    isolated = isolate_declared_holdout(
        BatteryDataset(full_frame), declaration=declaration,
    )
    require_isolated_training_dataset(isolated.development, isolated=isolated)
    expected = isolated.development.data[["ds", "microVolt"]].reset_index(drop=True)
    actual = processed_training.data[["ds", "microVolt"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert bool(processed_training.data["is_observed"].all())

    kwargs = inputs(processed_training)
    registry = ModelRegistry()
    registry.register("GRU", ProbeModel)
    kwargs["registry"] = registry
    assert kwargs["forecast_dates"] == list(declaration.forecast_dates)
    assert kwargs["training_context"]["train_end"] == declaration.development_cutoff
    assert kwargs["training_context"]["dataset_fingerprint"] == fingerprint_dataset(processed_training).to_dict()
    plan = build_internal_forecast_plan(**kwargs)
    assert verify_internal_forecast_plan(plan, **kwargs) is True
    execution = execute_internal_forecast_plan(
        plan, **kwargs, execution_authorized=True,
    )
    execution_artifact = build_internal_forecast_execution_artifact(
        execution, plan=plan, **kwargs,
    )
    assert verify_internal_forecast_execution_artifact(
        execution_artifact, plan=plan, **kwargs,
    ) is True
    contract = dict(
        declaration=declaration,
        selection_artifact=kwargs["artifact"],
        results=kwargs["results"],
        validation_context=kwargs["validation_context"],
        registry=kwargs["registry"],
        training_dataset=kwargs["training_dataset"],
        training_context=kwargs["training_context"],
        plan=plan,
        execution_artifact=execution_artifact,
        trusted_dataset_id="410",
    )
    binding = verify_holdout_binding(**contract)
    return isolated, contract, binding


def _reset_probe():
    ProbeModel.events = []
    ProbeModel.result_override = None
    ProbeModel.fail_at = None
    ProbeModel.mutate_training = False


def test_real_m18_execution_to_binding_to_b1_eligibility(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    isolated, contract, binding = _real_case(processed_training, full_frame, declaration)
    events_before = deepcopy(ProbeModel.events)
    eligibility = establish_holdout_evaluation_eligibility(
        **contract, binding_report=binding,
    )
    assert eligibility["status"] == "ELIGIBLE_FOR_HOLDOUT_DISCLOSURE"
    assert eligibility["forecast_dates"] == list(declaration.forecast_dates)
    assert eligibility["selected_model"] == "GRU"
    assert eligibility["target_values_included"] is False
    assert "actual_microVolt" not in eligibility
    assert "rmse" not in eligibility
    assert eligibility["execution_artifact_digest"] == binding["execution_artifact_digest"]
    assert eligibility["training_fingerprint"] == binding["training_fingerprint"]
    assert len(eligibility["predicted_microVolt"]) == len(isolated.holdout)
    assert ProbeModel.events == events_before
    assert [event[0] for event in events_before] == ["construct", "fit", "predict"]


def test_modified_execution_artifact_rejected_by_real_verifiers(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    _, contract, binding = _real_case(processed_training, full_frame, declaration)
    tampered = deepcopy(contract["execution_artifact"])
    tampered["execution_record"]["forecast"]["values_microVolt"][0] += 1.0
    verifier_kwargs = {
        "artifact": contract["selection_artifact"],
        "results": contract["results"],
        "validation_context": contract["validation_context"],
        "registry": contract["registry"],
        "training_dataset": contract["training_dataset"],
        "training_context": contract["training_context"],
        "forecast_horizon": len(declaration.forecast_dates),
        "forecast_dates": list(declaration.forecast_dates),
    }
    assert verify_internal_forecast_execution_artifact(
        tampered, plan=contract["plan"], **verifier_kwargs,
    ) is False
    with pytest.raises((ValueError, TypeError, AssertionError)):
        establish_holdout_evaluation_eligibility(
            **{**contract, "execution_artifact": tampered}, binding_report=binding,
        )


def test_holdout_target_changes_leave_real_b1_evidence_unchanged(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    isolated_a, contract_a, binding_a = _real_case(
        processed_training, full_frame, declaration,
    )
    evidence_a = establish_holdout_evaluation_eligibility(
        **contract_a, binding_report=binding_a,
    )
    changed = full_frame.copy(deep=True)
    changed.loc[
        changed["ds"] > pd.Timestamp(declaration.development_cutoff), "microVolt"
    ] = [999.0, -999.0]
    _reset_probe()
    isolated_b, contract_b, binding_b = _real_case(
        processed_training, changed, declaration,
    )
    evidence_b = establish_holdout_evaluation_eligibility(
        **contract_b, binding_report=binding_b,
    )
    assert isolated_a.holdout["microVolt"].tolist() != isolated_b.holdout["microVolt"].tolist()
    pd.testing.assert_frame_equal(isolated_a.development.data, isolated_b.development.data)
    assert contract_a["plan"] == contract_b["plan"]
    assert binding_a == binding_b
    assert evidence_a == evidence_b
    assert evidence_a["eligibility_digest"] == evidence_b["eligibility_digest"]
