"""M19-C2-A: independently constructed, individually valid B1-B4 chains.

These are synthetic contract fixtures, not independently authorized M18 runs.
They deliberately construct each alternative *before* B2/B3/B4/C1 issuance;
no accepted evidence is rewritten to manufacture an apparent valid chain.
"""
from copy import deepcopy

import pandas as pd
import pytest

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
    VerificationFinding, VerificationStatus,
)
from zenerestimation.assessment.holdout_isolation import isolate_declared_holdout
from zenerestimation.assessment.observability import (
    TargetObservation, TargetObservationEvidence, TargetOrigin,
    build_target_observability_report,
)
from zenerestimation.assessment.holdout_target_disclosure import (
    disclose_holdout_targets, _digest,
)
from zenerestimation.assessment.holdout_error_scoring import score_disclosed_holdout
from zenerestimation.assessment.holdout_evaluation_evidence import (
    build_holdout_evaluation_evidence, verify_holdout_evaluation_evidence,
)
from zenerestimation.comparison.benchmark_contract import (
    build_benchmark_entry, verify_benchmark_entry,
)
from zenerestimation.comparison.benchmark_comparability import (
    VerifiedBenchmarkInput, assess_scientific_alignment,
)
from zenerestimation.data.dataset import BatteryDataset


BASE_DATES = ("2024-03-01", "2024-06-01")
BASE_ACTUALS = (29.9, 30.506)
BASE_PREDICTIONS = (29.4, 29.5)


def _issue_chain(
    *, run_id, dataset_id="410", dates=BASE_DATES,
    actuals=BASE_ACTUALS, predictions=BASE_PREDICTIONS,
    model="GRU", family="GRU",
):
    """Issue a separate valid assessment and all four evidence stages."""
    assert len(dates) == len(actuals) == len(predictions)
    declaration = AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id=f"independent-{run_id}",
        assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id=dataset_id,
        development_cutoff="2023-12-01",
        forecast_dates=tuple(dates),
    )
    frame = pd.DataFrame({
        "ds": pd.to_datetime(["2023-09-01", "2023-12-01", *dates]),
        "microVolt": [29.2, 29.3, *actuals],
    })
    isolated = isolate_declared_holdout(BatteryDataset(frame), declaration=declaration)
    verified = VerificationFinding(
        status=VerificationStatus.VERIFIED,
        evidence_ref=f"measurement-log-{run_id}",
        verifier_id="fixture-verifier",
    )
    unknown = VerificationFinding(status=VerificationStatus.UNKNOWN)
    observations = tuple(
        TargetObservationEvidence(
            observation=TargetObservation(
                dataset_id=dataset_id, target_date=d, value_microVolt=v,
                origin=TargetOrigin.MEASURED,
                source_ref=f"measurement-log-{run_id}",
            ),
            provenance=verified, availability=unknown, exposure=unknown,
        )
        for d, v in zip(dates, actuals)
    )
    report = build_target_observability_report(declaration, observations)
    eligible = {
        "eligibility_version": "m19_b3_b1_evaluation_eligibility_v1",
        "status": "ELIGIBLE_FOR_HOLDOUT_DISCLOSURE",
        "assessment_id": declaration.assessment_id,
        "declaration_digest": declaration.digest,
        "dataset_id": dataset_id,
        "development_cutoff": "2023-12-01",
        "selected_model": model,
        "decision_digest": f"fixture-decision-{run_id}",
        "plan_digest": f"fixture-plan-{run_id}",
        "execution_artifact_digest": f"fixture-execution-{run_id}",
        "training_fingerprint": {"digest": f"fixture-training-{run_id}"},
        "forecast_horizon": len(dates),
        "forecast_dates": list(dates),
        "predicted_microVolt": list(predictions),
        "scientific_independence": "not_established",
        "target_values_included": False,
    }
    eligible["eligibility_digest"] = _digest(eligible)
    disclosure = disclose_holdout_targets(
        declaration=declaration, eligibility_snapshot=eligible,
        isolated=isolated, observability_report=report,
    )
    scoring = score_disclosed_holdout(
        declaration=declaration, eligibility_snapshot=eligible,
        disclosure_snapshot=disclosure,
    )
    context = dict(
        declaration=declaration,
        eligibility_snapshot=eligible,
        disclosure_snapshot=disclosure,
        scoring_snapshot=scoring,
    )
    evidence = build_holdout_evaluation_evidence(**context)
    assert verify_holdout_evaluation_evidence(evidence, **context)
    context["evaluation_evidence"] = evidence
    entry = build_benchmark_entry(
        run_id=run_id, model_family=family, **context,
    )
    assert verify_benchmark_entry(entry, **context)
    return VerifiedBenchmarkInput(entry=entry, **context)


def _pair(**changes):
    first = _issue_chain(run_id="case.GRU", model="GRU", family="GRU")
    second = _issue_chain(
        run_id="case.LSTM", model="LSTM", family="LSTM", **changes,
    )
    return [first, second]


def _assert_no_authorization(result):
    assert result["authorization"] == "not_established"
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False


def test_independently_issued_different_models_same_population_align():
    pair = _pair(predictions=(28.0, 31.0))
    assert pair[0].entry["assessment_id"] != pair[1].entry["assessment_id"]
    assert pair[0].entry["selected_model"] != pair[1].entry["selected_model"]
    assert pair[0].entry["metrics"] != pair[1].entry["metrics"]
    result = assess_scientific_alignment(pair)
    assert result["status"] == "SCIENTIFICALLY_ALIGNED"
    assert result["reasons"] == []
    assert result["shared_dataset_id"] == "410"
    assert result["shared_horizon"] == 2
    assert result["shared_forecast_dates"] == list(BASE_DATES)
    _assert_no_authorization(result)


@pytest.mark.parametrize("changes,reason", [
    ({"dataset_id": "110"}, "DATASET_MISMATCH"),
    ({"dates": ("2024-03-01", "2024-06-01", "2024-09-01"),
      "actuals": (29.9, 30.506, 30.707),
      "predictions": (29.4, 29.5, 29.6)}, "HORIZON_MISMATCH"),
    ({"dates": ("2024-06-01", "2024-09-01")}, "TIMESTAMP_MISMATCH"),
    ({"actuals": (29.9, 30.507)}, "OBSERVATION_MISMATCH"),
])
def test_independently_valid_mismatches_are_incomparable(changes, reason):
    pair = _pair(**changes)
    # Both chains are independently valid: no post-hoc artifact mutation.
    for item in pair:
        assert verify_benchmark_entry(
            item.entry, declaration=item.declaration,
            eligibility_snapshot=item.eligibility_snapshot,
            disclosure_snapshot=item.disclosure_snapshot,
            scoring_snapshot=item.scoring_snapshot,
            evaluation_evidence=item.evaluation_evidence,
        )
    result = assess_scientific_alignment(pair)
    assert result["status"] == "INCOMPARABLE"
    assert reason in result["reasons"]
    assert result["observation_digest"] is None
    _assert_no_authorization(result)


def test_alignment_is_order_independent_and_does_not_mutate_evidence():
    pair = _pair(predictions=(28.0, 31.0))
    before = [deepcopy(item.entry) for item in pair]
    first = assess_scientific_alignment(pair)
    second = assess_scientific_alignment(pair[::-1])
    assert first == second
    assert [item.entry for item in pair] == before
    _assert_no_authorization(first)


def test_predictions_change_metrics_but_not_observation_digest():
    baseline = assess_scientific_alignment(_pair())
    alternative = assess_scientific_alignment(_pair(predictions=(28.0, 31.0)))
    assert baseline["status"] == alternative["status"] == "SCIENTIFICALLY_ALIGNED"
    assert baseline["observation_digest"] == alternative["observation_digest"]
    _assert_no_authorization(alternative)
