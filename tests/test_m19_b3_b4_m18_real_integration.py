"""M19-B3-B4: genuine M18 -> B1 -> B2 -> B3 -> B4 integration.

Uses the accepted deterministic M18 ProbeModel fixture. Verification findings
are fixture-supplied and are not external authentication.
"""
from copy import deepcopy
from math import fsum, sqrt

import pandas as pd
import pytest

from tests.test_m18_forecast_plan_contract import training as processed_training
from tests.test_m18_forecast_execution_contract import ProbeModel
from tests.test_m19_b3_b1_m18_real_integration import declaration, full_frame, _reset_probe
from tests.test_m19_b3_b3_m18_real_integration import _pipeline
from zenerestimation.assessment.holdout_evaluation_evidence import (
    build_holdout_evaluation_evidence,
    verify_holdout_evaluation_evidence,
)
from zenerestimation.assessment.holdout_target_disclosure import _digest


def _complete(processed_training, frame, declaration):
    eligibility, disclosure, scoring = _pipeline(processed_training, frame, declaration)
    events = deepcopy(ProbeModel.events)
    evidence = build_holdout_evaluation_evidence(
        declaration=declaration, eligibility_snapshot=eligibility,
        disclosure_snapshot=disclosure, scoring_snapshot=scoring,
    )
    assert verify_holdout_evaluation_evidence(
        evidence, declaration=declaration, eligibility_snapshot=eligibility,
        disclosure_snapshot=disclosure, scoring_snapshot=scoring,
    )
    assert ProbeModel.events == events
    return eligibility, disclosure, scoring, evidence


def test_real_m18_to_b4_reconstructs_evaluation_evidence(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    eligibility, disclosure, scoring, evidence = _complete(
        processed_training, full_frame, declaration,
    )
    assert evidence["status"] == "HOLDOUT_EVALUATION_EVIDENCE_VERIFIED"
    assert evidence["scientific_independence"] == "not_established"
    assert evidence["authentication"] == "not_established"
    assert evidence["eligibility_digest"] == eligibility["eligibility_digest"]
    assert evidence["disclosure_digest"] == disclosure["disclosure_digest"]
    assert evidence["scoring_digest"] == scoring["scoring_digest"]
    assert evidence["execution_artifact_digest"] == eligibility["execution_artifact_digest"]
    assert evidence["forecast_dates"] == list(declaration.forecast_dates)
    assert evidence["by_date"] == scoring["by_date"]
    errors = [
        float(p) - float(a)
        for p, a in zip(eligibility["predicted_microVolt"], disclosure["observed_microVolt"])
    ]
    n = len(errors)
    assert evidence["metrics"]["rmse_microVolt"] == pytest.approx(sqrt(fsum(e*e for e in errors)/n))
    assert evidence["metrics"]["mae_microVolt"] == pytest.approx(fsum(abs(e) for e in errors)/n)
    assert evidence["metrics"]["bias_microVolt"] == pytest.approx(fsum(errors)/n)
    assert evidence["metrics"]["mape_percent"] == pytest.approx(
        fsum(abs(e/a)*100 for e, a in zip(errors, disclosure["observed_microVolt"]))/n
    )
    assert evidence == build_holdout_evaluation_evidence(
        declaration=declaration, eligibility_snapshot=eligibility,
        disclosure_snapshot=disclosure, scoring_snapshot=scoring,
    )


def test_holdout_perturbation_changes_b4_evidence_not_m18_forecast(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    eligible_a, disclosed_a, scored_a, evidence_a = _complete(
        processed_training, full_frame, declaration,
    )
    changed = full_frame.copy(deep=True)
    changed.loc[
        changed["ds"] > pd.Timestamp(declaration.development_cutoff), "microVolt"
    ] = [31.9, 32.506]
    _reset_probe()
    eligible_b, disclosed_b, scored_b, evidence_b = _complete(
        processed_training, changed, declaration,
    )
    assert eligible_a == eligible_b
    assert evidence_a["eligibility_digest"] == evidence_b["eligibility_digest"]
    assert evidence_a["execution_artifact_digest"] == evidence_b["execution_artifact_digest"]
    assert disclosed_a["disclosure_digest"] != disclosed_b["disclosure_digest"]
    assert scored_a["scoring_digest"] != scored_b["scoring_digest"]
    assert evidence_a["evaluation_evidence_digest"] != evidence_b["evaluation_evidence_digest"]
    assert evidence_a["metrics"]["bias_microVolt"] - evidence_b["metrics"]["bias_microVolt"] == pytest.approx(2.0)
    for a, b in zip(evidence_a["by_date"], evidence_b["by_date"]):
        assert a["forecast_microVolt"] == b["forecast_microVolt"]
        assert a["signed_error_microVolt"] - b["signed_error_microVolt"] == pytest.approx(2.0)


def test_real_b4_rejects_redigested_scoring_and_evidence_tampering(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    eligibility, disclosure, scoring, evidence = _complete(
        processed_training, full_frame, declaration,
    )
    events = deepcopy(ProbeModel.events)
    forged_scoring = deepcopy(scoring)
    forged_scoring["metrics"]["rmse_microVolt"] += 10.0
    forged_scoring["scoring_digest"] = _digest({
        key: value for key, value in forged_scoring.items() if key != "scoring_digest"
    })
    with pytest.raises(ValueError, match="scoring differs"):
        build_holdout_evaluation_evidence(
            declaration=declaration, eligibility_snapshot=eligibility,
            disclosure_snapshot=disclosure, scoring_snapshot=forged_scoring,
        )
    assert not verify_holdout_evaluation_evidence(
        evidence, declaration=declaration, eligibility_snapshot=eligibility,
        disclosure_snapshot=disclosure, scoring_snapshot=forged_scoring,
    )
    forged_evidence = deepcopy(evidence)
    forged_evidence["by_date"][0]["signed_error_microVolt"] += 1.0
    forged_evidence["evaluation_evidence_digest"] = _digest({
        key: value for key, value in forged_evidence.items() if key != "evaluation_evidence_digest"
    })
    assert not verify_holdout_evaluation_evidence(
        forged_evidence, declaration=declaration, eligibility_snapshot=eligibility,
        disclosure_snapshot=disclosure, scoring_snapshot=scoring,
    )
    assert ProbeModel.events == events
