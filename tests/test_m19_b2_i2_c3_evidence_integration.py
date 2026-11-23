"""M19-B2-I2-C3 contract tests: cross-contract integration, not certification.

The C1 fixture uses an isolated M18 verifier seam. Real M18 integration belongs
in a separate test file and is not claimed by this suite.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from tests.test_m19_b2_i2_c1_alignment import case as c1_case
from zenerestimation.assessment.eligibility import (
    VerificationFinding, VerificationStatus,
)
from zenerestimation.assessment.temporal import (
    TEMPORAL_CONTRACT_VERSION, TemporalEligibilityEvidence,
)
from zenerestimation.assessment.observability import (
    TargetObservation, TargetObservationEvidence, TargetOrigin,
    build_target_observability_report,
)
from zenerestimation.assessment.observed_scoring import align_verified_observations
from zenerestimation.assessment.observed_metrics import score_verified_observations
from zenerestimation.assessment.evidence_integration import (
    INTEGRATION_CONTRACT_VERSION, IntegratedAssessmentEvidence,
    TemporalSummaryStatus, integrate_assessment_evidence,
)


TEMPORAL_FIELDS = (
    "plan_binding", "execution_integrity", "training_cutoff",
    "development_boundary", "forecast_freeze", "target_availability",
    "chronology",
)


def finding(status):
    if status is VerificationStatus.VERIFIED:
        return VerificationFinding(status, "evidence.ref", "trusted.verifier")
    if status is VerificationStatus.FAILED:
        return VerificationFinding(status, reason_codes=("EVIDENCE_FAILED",))
    return VerificationFinding(status)


def temporal_for(declaration, statuses=None):
    statuses = statuses or {}
    values = {name: finding(statuses.get(name, VerificationStatus.VERIFIED))
              for name in TEMPORAL_FIELDS}
    return TemporalEligibilityEvidence(
        assessment_id=declaration.assessment_id,
        declaration_digest=declaration.digest,
        contract_version=TEMPORAL_CONTRACT_VERSION,
        **values,
    )


@pytest.fixture
def integrated_args(c1_case):
    c1_args, _ = c1_case
    declaration = c1_args["declaration"]
    alignment = align_verified_observations(**c1_args)
    return dict(
        declaration=declaration,
        temporal_evidence=temporal_for(declaration),
        observability_report=c1_args["observability_report"],
        alignment_result=alignment,
        metric_result=score_verified_observations(alignment),
    )


def rebind_observations(args, observations):
    """Build a consistent B/C1/C2 trio without re-verifying M18."""
    report = build_target_observability_report(args["declaration"], observations)
    old_alignment = args["alignment_result"]
    rows = []
    for old, evidence in zip(old_alignment.rows, report.observations):
        eligible = evidence.verified_observed
        reasons = () if eligible else (
            ("ORIGIN_NOT_MEASURED",) if evidence.observation.origin is not TargetOrigin.MEASURED
            else ("PROVENANCE_NOT_VERIFIED",)
        )
        rows.append(replace(
            old,
            observed_microVolt=float(evidence.observation.value_microVolt) if eligible else None,
            eligible=eligible,
            exclusion_reasons=reasons,
            observation_evidence=evidence,
        ))
    alignment = replace(old_alignment, rows=tuple(rows))
    return dict(args, observability_report=report, alignment_result=alignment,
                metric_result=score_verified_observations(alignment))


def test_c3_01_partial_coverage_integrates_and_preserves_evidence(integrated_args):
    result = integrate_assessment_evidence(**integrated_args)
    payload = result.to_dict()
    assert isinstance(result, IntegratedAssessmentEvidence)
    assert result.contract_version == INTEGRATION_CONTRACT_VERSION
    assert result.temporal_status is TemporalSummaryStatus.VERIFIED_FINDINGS
    assert result.coverage_status == "partial"
    assert result.score_available is True
    assert result.declaration_digest == integrated_args["declaration"].digest
    assert result.execution_artifact_digest == integrated_args["alignment_result"].execution_artifact_digest
    assert payload["evidence"]["observability"]["coverage_mask"] == [True, False, False]
    assert payload["evidence"]["observed_metrics"]["eligible_count"] == 1
    assert "certified" not in json.dumps(payload).lower()
    assert "scientific_classification" not in payload


def test_c3_02_complete_coverage(integrated_args):
    original = integrated_args["observability_report"].observations
    verified = finding(VerificationStatus.VERIFIED)
    observations = tuple(
        replace(e, provenance=verified,
                observation=replace(e.observation,
                    origin=TargetOrigin.MEASURED, value_microVolt=31.0,
                    source_ref="source.ref"))
        for e in original
    )
    args = rebind_observations(integrated_args, observations)
    result = integrate_assessment_evidence(**args)
    assert result.coverage_status == "complete"
    assert result.metric_result.eligible_count == 3
    assert result.score_available is True


def test_c3_03_zero_coverage_is_unavailable_not_zero(integrated_args):
    original = integrated_args["observability_report"].observations
    unknown = finding(VerificationStatus.UNKNOWN)
    observations = tuple(
        TargetObservationEvidence(
            TargetObservation(e.observation.dataset_id, e.observation.target_date,
                              None, TargetOrigin.MISSING),
            unknown, unknown, unknown,
        )
        for e in original
    )
    args = rebind_observations(integrated_args, observations)
    result = integrate_assessment_evidence(**args)
    assert result.coverage_status == "none"
    assert result.score_available is False
    metrics = result.to_dict()["evidence"]["observed_metrics"]
    assert metrics["metrics"] == {
        "rmse_microVolt": None, "mae_microVolt": None, "mape_percent": None
    }
    assert metrics["denominators"] == {"rmse": 0, "mae": 0, "mape": 0}


@pytest.mark.parametrize("field", ["temporal_evidence", "observability_report",
                                     "alignment_result", "metric_result"])
def test_c3_04_assessment_identity_mismatch(integrated_args, field):
    args = dict(integrated_args)
    args[field] = replace(args[field], assessment_id="other.assessment")
    with pytest.raises(ValueError, match="assessment_id"):
        integrate_assessment_evidence(**args)


@pytest.mark.parametrize("field", ["observability_report", "alignment_result",
                                     "metric_result"])
def test_c3_05_dataset_identity_mismatch(integrated_args, field):
    args = dict(integrated_args)
    if field == "observability_report":
        report = args[field]
        observations = tuple(
            replace(item, observation=replace(item.observation, dataset_id="other.dataset"))
            for item in report.observations
        )
        args[field] = replace(report, dataset_id="other.dataset", observations=observations)
    else:
        args[field] = replace(args[field], dataset_id="other.dataset")
    with pytest.raises(ValueError, match="dataset_id"):
        integrate_assessment_evidence(**args)


@pytest.mark.parametrize("field", ["temporal_evidence", "observability_report",
                                     "alignment_result", "metric_result"])
def test_c3_06_declaration_digest_mismatch(integrated_args, field):
    args = dict(integrated_args)
    args[field] = replace(args[field], declaration_digest="sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="declaration_digest"):
        integrate_assessment_evidence(**args)


@pytest.mark.parametrize("field", ["temporal_evidence", "observability_report",
                                     "alignment_result", "metric_result"])
def test_c3_07_wrong_contract_version_rejected(integrated_args, field):
    args = dict(integrated_args)
    if field in ("temporal_evidence", "observability_report"):
        # These upstream immutable contracts reject malformed versions during
        # construction, before a C3 integration call is possible.
        with pytest.raises(ValueError, match="contract version"):
            replace(args[field], contract_version="WRONG_VERSION")
    else:
        # C1/C2 result types permit construction with an unsupported version;
        # C3 must reject the inconsistent contract at its own boundary.
        args[field] = replace(args[field], contract_version="WRONG_VERSION")
        with pytest.raises(ValueError, match="contract"):
            integrate_assessment_evidence(**args)


@pytest.mark.parametrize("field", ["temporal_evidence", "observability_report",
                                     "alignment_result", "metric_result"])
def test_c3_08_untrusted_serialized_inputs_rejected(integrated_args, field):
    args = dict(integrated_args)
    args[field] = args[field].to_dict()
    with pytest.raises(TypeError, match=field):
        integrate_assessment_evidence(**args)


def test_c3_09_wrong_observation_calendar_rejected(integrated_args):
    args = dict(integrated_args)
    report = args["observability_report"]
    # Same horizon, different valid quarterly dates: constructor accepts it,
    # declaration-bound builder must reject it.
    shifted = tuple(
        replace(e, observation=replace(e.observation,
            target_date=("2025-06-01", "2025-09-01", "2025-12-01")[i]))
        for i, e in enumerate(report.observations)
    )
    args["observability_report"] = replace(report, observations=shifted)
    with pytest.raises(ValueError, match="dates"):
        integrate_assessment_evidence(**args)


def test_c3_10_wrong_alignment_calendar_rejected(integrated_args):
    args = dict(integrated_args)
    alignment = args["alignment_result"]
    shifted = replace(alignment.rows[0], target_date="2025-12-01",
                      observation_evidence=replace(
                          alignment.rows[0].observation_evidence,
                          observation=replace(
                              alignment.rows[0].observation_evidence.observation,
                              target_date="2025-12-01")))
    args["alignment_result"] = replace(alignment, rows=(shifted, *alignment.rows[1:]))
    with pytest.raises(ValueError, match="calendar"):
        integrate_assessment_evidence(**args)


def test_c3_11_wrong_metric_horizon_rejected(integrated_args):
    args = dict(integrated_args)
    args["metric_result"] = replace(args["metric_result"], declared_horizon=4)
    with pytest.raises(ValueError, match="horizon"):
        integrate_assessment_evidence(**args)


def test_c3_12_forged_alignment_coverage_rejected(integrated_args):
    args = dict(integrated_args)
    alignment = args["alignment_result"]
    row = alignment.rows[1]
    args["alignment_result"] = replace(
        alignment, rows=(alignment.rows[0],
                         replace(row, eligible=True, observed_microVolt=31.0,
                                 exclusion_reasons=()),
                         alignment.rows[2]))
    with pytest.raises(ValueError, match="coverage|eligibility"):
        integrate_assessment_evidence(**args)


def test_c3_13_mismatched_observation_evidence_rejected(integrated_args):
    args = dict(integrated_args)
    alignment = args["alignment_result"]
    first = alignment.rows[0]
    altered = replace(first.observation_evidence,
                      observation=replace(first.observation_evidence.observation,
                                          value_microVolt=32.0))
    args["alignment_result"] = replace(
        alignment, rows=(replace(first, observation_evidence=altered,
                                 observed_microVolt=32.0), *alignment.rows[1:]))
    with pytest.raises(ValueError, match="observation evidence"):
        integrate_assessment_evidence(**args)


def test_c3_14_execution_digest_mismatch_rejected(integrated_args):
    args = dict(integrated_args)
    args["metric_result"] = replace(args["metric_result"],
                                    execution_artifact_digest="sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="execution artifact digest"):
        integrate_assessment_evidence(**args)


@pytest.mark.parametrize("change", [
    {"rmse_microVolt": 123.0},
    {"mae_microVolt": 123.0},
    {"mape_percent": 123.0},
    {"rmse_denominator": 99},
    {"mape_denominator": 99},
    {"eligible_count": 99},
    {"coverage_mask": (False, False, False)},
    {"mape_zero_policy": "INCLUDE_ZERO_ACTUALS"},
])
def test_c3_15_forged_metric_rejected(integrated_args, change):
    args = dict(integrated_args)
    args["metric_result"] = replace(args["metric_result"], **change)
    with pytest.raises(ValueError, match="observed metric result"):
        integrate_assessment_evidence(**args)


def test_c3_16_forged_metric_alignment_rows_rejected(integrated_args):
    args = dict(integrated_args)
    metric = args["metric_result"]
    rows = deepcopy(metric.alignment_rows)
    rows[0]["forecast_microVolt"] = 999.0
    args["metric_result"] = replace(metric, alignment_rows=tuple(rows))
    with pytest.raises(ValueError, match="observed metric result"):
        integrate_assessment_evidence(**args)


@pytest.mark.parametrize("status,expected", [
    (VerificationStatus.FAILED, TemporalSummaryStatus.FAILED),
    (VerificationStatus.UNKNOWN, TemporalSummaryStatus.INCOMPLETE),
    (VerificationStatus.VERIFIED, TemporalSummaryStatus.VERIFIED_FINDINGS),
])
def test_c3_17_temporal_status_preserved(integrated_args, status, expected):
    args = dict(integrated_args)
    args["temporal_evidence"] = temporal_for(
        args["declaration"], {"chronology": status})
    result = integrate_assessment_evidence(**args)
    assert result.temporal_status is expected
    assert result.score_available is True
    assert result.to_dict()["evidence"]["temporal"]["chronology"]["status"] == status.value
    assert "certified" not in json.dumps(result.to_dict()).lower()


def test_c3_18_failure_precedes_unknown(integrated_args):
    args = dict(integrated_args)
    args["temporal_evidence"] = temporal_for(args["declaration"], {
        "chronology": VerificationStatus.UNKNOWN,
        "forecast_freeze": VerificationStatus.FAILED,
    })
    assert integrate_assessment_evidence(**args).temporal_status is TemporalSummaryStatus.FAILED


def test_c3_19_serialization_is_deterministic_and_detached(integrated_args):
    a = integrate_assessment_evidence(**integrated_args)
    b = integrate_assessment_evidence(**integrated_args)
    original = a.to_dict()
    assert original == b.to_dict()
    assert json.dumps(original, sort_keys=True, allow_nan=False) == json.dumps(
        b.to_dict(), sort_keys=True, allow_nan=False)
    altered = a.to_dict()
    altered["evidence"]["observed_metrics"]["alignment_rows"][0]["forecast_microVolt"] = -1
    altered["evidence"]["observability"]["coverage_mask"][0] = False
    assert a.to_dict() == original


def test_c3_20_snapshot_isolated_from_nested_source_mutation(integrated_args):
    result = integrate_assessment_evidence(**integrated_args)
    before = result.to_dict()
    # The existing C2 type contains nested mutable dictionaries; C3's
    # serialized snapshot must remain stable even if source aliases mutate.
    integrated_args["metric_result"].alignment_rows[0]["forecast_microVolt"] = -99
    assert result.to_dict() == before


def test_c3_21_frozen_result_fields(integrated_args):
    result = integrate_assessment_evidence(**integrated_args)
    with pytest.raises(FrozenInstanceError):
        result.assessment_id = "changed"
    with pytest.raises(FrozenInstanceError):
        result.score_available = False
