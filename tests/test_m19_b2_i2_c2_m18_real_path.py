"""M19-B2-I2-C2 real production path: M18 artifact -> C1 -> C2.

Synthetic evidence supplied by fixtures is not independently authenticated.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

# Explicitly register the transitive pytest fixtures when importing fixtures
# from another test module (pytest does not auto-register their dependencies).
from tests.test_m18_forecast_plan_contract import training as training_fixture
from tests.test_m18_forecast_execution_contract import prepared as execution_prepared
from tests.test_m19_b2_i2_c1_m18_integration import (
    integrated_case,
    ProbeModel,
)
from zenerestimation.assessment.observed_scoring import align_verified_observations
from zenerestimation.assessment.observed_metrics import (
    MAPE_ZERO_POLICY,
    SCORING_CONTRACT_VERSION,
    score_verified_observations,
)


def _alignment_args(case):
    # C1 derives the horizon and calendar from the declaration, not caller hints.
    return {k: v for k, v in case.items()
            if k not in ("forecast_horizon", "forecast_dates")}


def test_real_m18_c1_c2_partial_coverage_and_exact_metrics(integrated_case):
    args = integrated_case
    alignment = align_verified_observations(**_alignment_args(args))
    scored = score_verified_observations(alignment)

    assert scored.contract_version == SCORING_CONTRACT_VERSION
    assert scored.alignment_contract_version == alignment.contract_version
    assert scored.assessment_id == alignment.assessment_id
    assert scored.dataset_id == alignment.dataset_id
    assert scored.declaration_digest == alignment.declaration_digest
    assert scored.execution_artifact_digest == args["execution_artifact"]["artifact_digest"]

    assert scored.declared_horizon == 2
    assert scored.coverage_mask == (True, False)
    assert scored.eligible_count == 1
    assert scored.excluded_count == 1
    assert scored.eligible_dates == ("2024-03-01",)
    assert scored.excluded_dates == ("2024-06-01",)
    assert scored.score_available is True
    assert scored.rmse_denominator == 1
    assert scored.mae_denominator == 1
    assert scored.mape_denominator == 1
    assert scored.rmse_microVolt == pytest.approx(0.5)
    assert scored.mae_microVolt == pytest.approx(0.5)
    assert scored.mape_percent == pytest.approx(100 * 0.5 / 29.9)
    assert scored.mape_zero_policy == MAPE_ZERO_POLICY
    assert scored.mape_zero_actual_excluded_dates == ()
    assert len(scored.alignment_rows) == 2
    assert scored.alignment_rows[1]["eligible"] is False
    assert scored.alignment_rows[1]["observed_microVolt"] is None
    assert "ORIGIN_NOT_MEASURED" in scored.alignment_rows[1]["exclusion_reasons"]
    assert scored.to_dict()["execution_artifact_digest"] == alignment.execution_artifact_digest
    assert ProbeModel.events == []  # Neither verifier, alignment nor scoring reruns model.


def test_real_path_rejects_tampered_execution_before_scoring(integrated_case):
    args = _alignment_args(integrated_case)
    args["execution_artifact"] = deepcopy(args["execution_artifact"])
    args["execution_artifact"]["execution_record"]["forecast"]["values_microVolt"][0] = 999.0
    with pytest.raises(ValueError, match="execution artifact verification failed"):
        score_verified_observations(align_verified_observations(**args))
    assert ProbeModel.events == []


def test_real_path_rejects_changed_training_context_before_scoring(integrated_case):
    args = _alignment_args(integrated_case)
    args["training_context"] = deepcopy(args["training_context"])
    args["training_context"]["dataset_fingerprint"]["digest"] = "0" * 64
    with pytest.raises(ValueError, match="execution artifact verification failed"):
        score_verified_observations(align_verified_observations(**args))
    assert ProbeModel.events == []
