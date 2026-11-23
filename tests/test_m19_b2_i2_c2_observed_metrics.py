"""M19-B2-I2-C2 observed-only scoring contract tests."""
from dataclasses import replace
import pytest

from tests.test_m19_b2_i2_c1_alignment import case as c1_case
from zenerestimation.assessment.observed_scoring import align_verified_observations
from zenerestimation.assessment.observed_metrics import score_verified_observations


@pytest.fixture
def alignment(c1_case):
    args, _ = c1_case
    return align_verified_observations(**args)


def test_partial_coverage_only_scores_verified_observations(alignment):
    result = score_verified_observations(alignment)
    assert result.declared_horizon == 3
    assert (result.eligible_count, result.excluded_count) == (1, 2)
    assert result.coverage_mask == (True, False, False)
    assert result.rmse_microVolt == pytest.approx(1.0)
    assert result.mae_microVolt == pytest.approx(1.0)
    assert result.mape_percent == pytest.approx(100.0 / 31.0)
    assert (result.rmse_denominator, result.mae_denominator, result.mape_denominator) == (1, 1, 1)
    assert result.to_dict()["alignment_rows"][1]["exclusion_reasons"] == ["PROVENANCE_NOT_VERIFIED"]


def test_zero_coverage_returns_unavailable_not_zero(alignment):
    rows = tuple(replace(row, eligible=False, observed_microVolt=None,
                         exclusion_reasons=("PROVENANCE_NOT_VERIFIED",),
                         observation_evidence=replace(row.observation_evidence,
                             provenance=alignment.rows[1].observation_evidence.provenance))
                 for row in alignment.rows)
    result = score_verified_observations(replace(alignment, rows=rows))
    assert result.score_available is False
    assert result.rmse_microVolt is None
    assert result.mae_microVolt is None
    assert result.mape_percent is None
    assert result.rmse_denominator == result.mae_denominator == result.mape_denominator == 0


def test_zero_actual_is_excluded_from_mape_but_not_rmse_mae(alignment):
    row = alignment.rows[0]
    obs = replace(row.observation_evidence.observation, value_microVolt=0.0)
    evidence = replace(row.observation_evidence, observation=obs)
    changed = replace(row, observed_microVolt=0.0, observation_evidence=evidence)
    result = score_verified_observations(replace(alignment, rows=(changed, *alignment.rows[1:])))
    assert result.rmse_microVolt == pytest.approx(30.0)
    assert result.mae_microVolt == pytest.approx(30.0)
    assert result.mape_percent is None
    assert result.mape_denominator == 0
    assert result.mape_zero_actual_excluded_dates == (row.target_date,)


def test_rejects_untrusted_serialized_input(alignment):
    with pytest.raises(TypeError):
        score_verified_observations(alignment.to_dict())


def test_rejects_forged_eligibility(alignment):
    row = alignment.rows[1]
    forged = replace(row, eligible=True, observed_microVolt=31.0, exclusion_reasons=())
    with pytest.raises(ValueError, match="eligibility"):
        score_verified_observations(replace(alignment, rows=(alignment.rows[0], forged, alignment.rows[2])))


def test_rejects_forged_observed_value(alignment):
    row = alignment.rows[0]
    forged = replace(row, observed_microVolt=999.0)
    with pytest.raises(ValueError, match="verified measurement"):
        score_verified_observations(replace(alignment, rows=(forged, *alignment.rows[1:])))


def test_rejects_duplicate_dates(alignment):
    duplicated = replace(alignment.rows[1], target_date=alignment.rows[0].target_date,
                         observation_evidence=alignment.rows[0].observation_evidence,
                         eligible=True, observed_microVolt=31.0, exclusion_reasons=())
    with pytest.raises(ValueError, match="dates"):
        score_verified_observations(replace(alignment, rows=(alignment.rows[0], duplicated, alignment.rows[2])))


def test_deterministic_serialization_and_upstream_binding(alignment):
    a = score_verified_observations(alignment)
    b = score_verified_observations(alignment)
    assert a.to_dict() == b.to_dict()
    assert a.declaration_digest == alignment.declaration_digest
    assert a.execution_artifact_digest == alignment.execution_artifact_digest
    assert a.assessment_id == alignment.assessment_id
    assert a.dataset_id == alignment.dataset_id
