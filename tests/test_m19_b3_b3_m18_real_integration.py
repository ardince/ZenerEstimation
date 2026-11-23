"""M19-B3-B3 genuine M18 -> B1 -> B2 -> B3 integration tests.

Uses the existing deterministic ProbeModel and real M18 planning, execution,
serialization, verification, and B1/B2 contracts. Fixture provenance findings
are supplied assertions, not external authentication or independence proof.
"""
from copy import deepcopy
from math import fsum, isclose, sqrt

import pandas as pd
import pytest

from tests.test_m18_forecast_plan_contract import training as processed_training
from tests.test_m18_forecast_execution_contract import ProbeModel
from tests.test_m19_b3_b1_m18_real_integration import (
    declaration, full_frame, _reset_probe,
)
from tests.test_m19_b3_b2_m18_real_integration import _case, _report
from zenerestimation.assessment.holdout_target_disclosure import (
    disclose_holdout_targets, _digest,
)
from zenerestimation.assessment.holdout_error_scoring import score_disclosed_holdout


def _pipeline(processed_training, frame, declaration):
    """Build genuine upstream evidence, then disclose and score without rerunning M18."""
    isolated, eligibility = _case(processed_training, frame, declaration)
    events_after_execution = deepcopy(ProbeModel.events)
    assert [event[0] for event in events_after_execution] == [
        "construct", "fit", "predict",
    ]
    disclosure = disclose_holdout_targets(
        declaration=declaration,
        eligibility_snapshot=eligibility,
        isolated=isolated,
        observability_report=_report(declaration, isolated),
    )
    scoring = score_disclosed_holdout(
        declaration=declaration,
        eligibility_snapshot=eligibility,
        disclosure_snapshot=disclosure,
    )
    assert ProbeModel.events == events_after_execution
    return eligibility, disclosure, scoring


def test_real_m18_to_b3_independent_error_reconstruction(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    eligibility, disclosure, scored = _pipeline(
        processed_training, full_frame, declaration,
    )
    predictions = eligibility["predicted_microVolt"]
    actuals = [29.9, 30.506]
    errors = [float(prediction) - actual for prediction, actual in zip(predictions, actuals)]
    assert disclosure["observed_microVolt"] == actuals
    assert scored["status"] == "HOLDOUT_SCORED"
    assert scored["forecast_dates"] == list(declaration.forecast_dates)
    assert scored["eligibility_digest"] == eligibility["eligibility_digest"]
    assert scored["disclosure_digest"] == disclosure["disclosure_digest"]
    assert scored["execution_artifact_digest"] == eligibility["execution_artifact_digest"]
    assert scored["error_convention"] == "forecast_minus_observed"
    assert scored["scored_count"] == len(actuals)
    assert scored["denominators"] == {"rmse": 2, "mae": 2, "bias": 2, "mape": 2}
    assert [row["target_date"] for row in scored["by_date"]] == list(declaration.forecast_dates)
    for row, prediction, actual, error in zip(scored["by_date"], predictions, actuals, errors):
        assert row["forecast_microVolt"] == pytest.approx(prediction)
        assert row["observed_microVolt"] == pytest.approx(actual)
        assert row["signed_error_microVolt"] == pytest.approx(error)
        assert row["absolute_error_microVolt"] == pytest.approx(abs(error))
        assert row["squared_error_microVolt2"] == pytest.approx(error * error)
    assert scored["metrics"]["rmse_microVolt"] == pytest.approx(sqrt(fsum(e*e for e in errors)/2))
    assert scored["metrics"]["mae_microVolt"] == pytest.approx(fsum(abs(e) for e in errors)/2)
    assert scored["metrics"]["bias_microVolt"] == pytest.approx(fsum(errors)/2)
    assert scored["metrics"]["mape_percent"] == pytest.approx(
        fsum(abs(e/a)*100 for e, a in zip(errors, actuals))/2
    )
    assert scored == score_disclosed_holdout(
        declaration=declaration, eligibility_snapshot=eligibility,
        disclosure_snapshot=disclosure,
    )


def test_holdout_perturbation_preserves_real_forecast_but_changes_b3_scores(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    eligible_a, disclosed_a, scored_a = _pipeline(processed_training, full_frame, declaration)
    changed = full_frame.copy(deep=True)
    changed.loc[
        changed["ds"] > pd.Timestamp(declaration.development_cutoff), "microVolt"
    ] = [31.9, 32.506]
    _reset_probe()
    eligible_b, disclosed_b, scored_b = _pipeline(processed_training, changed, declaration)
    assert eligible_a == eligible_b
    assert eligible_a["predicted_microVolt"] == eligible_b["predicted_microVolt"]
    assert disclosed_a["observed_microVolt"] == [29.9, 30.506]
    assert disclosed_b["observed_microVolt"] == [31.9, 32.506]
    assert disclosed_a["disclosure_digest"] != disclosed_b["disclosure_digest"]
    assert scored_a["eligibility_digest"] == scored_b["eligibility_digest"]
    assert scored_a["disclosure_digest"] != scored_b["disclosure_digest"]
    assert scored_a["scoring_digest"] != scored_b["scoring_digest"]
    assert scored_a["metrics"]["bias_microVolt"] - scored_b["metrics"]["bias_microVolt"] == pytest.approx(2.0)
    for row_a, row_b in zip(scored_a["by_date"], scored_b["by_date"]):
        assert row_a["forecast_microVolt"] == row_b["forecast_microVolt"]
        assert row_a["signed_error_microVolt"] - row_b["signed_error_microVolt"] == pytest.approx(2.0)
    assert not isclose(scored_a["metrics"]["mae_microVolt"], scored_b["metrics"]["mae_microVolt"])


def test_real_eligibility_cannot_score_inconsistent_redigested_disclosure(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    eligible, disclosed, scored = _pipeline(processed_training, full_frame, declaration)
    events = deepcopy(ProbeModel.events)
    tampered = deepcopy(disclosed)
    tampered["observed_microVolt"][0] += 5.0
    tampered["disclosure_digest"] = _digest({
        key: value for key, value in tampered.items() if key != "disclosure_digest"
    })
    with pytest.raises(ValueError, match="observation"):
        score_disclosed_holdout(
            declaration=declaration, eligibility_snapshot=eligible,
            disclosure_snapshot=tampered,
        )
    assert ProbeModel.events == events
    assert disclosed["observed_microVolt"] == [29.9, 30.506]
    assert scored["status"] == "HOLDOUT_SCORED"
