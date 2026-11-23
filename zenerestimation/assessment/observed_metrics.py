"""M19-B2-I2-C2: observed-only accuracy metrics from verified C1 alignment.

This scorer does not verify M18 artifacts itself: it accepts only the in-memory
C1 alignment result, never an untrusted serialized dict. It makes no claim of
independent validation or external measurement authentication.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt, fsum

from zenerestimation.assessment.observed_scoring import (
    ALIGNMENT_CONTRACT_VERSION, ObservedAlignmentResult,
)

SCORING_CONTRACT_VERSION = "M19_B2_I2_C2_OBSERVED_METRICS_V1"
MAPE_ZERO_POLICY = "EXCLUDE_ZERO_ACTUALS"


@dataclass(frozen=True, slots=True)
class ObservedMetricResult:
    contract_version: str
    assessment_id: str
    dataset_id: str
    declaration_digest: str
    execution_artifact_digest: str
    alignment_contract_version: str
    declared_horizon: int
    eligible_count: int
    excluded_count: int
    eligible_dates: tuple[str, ...]
    excluded_dates: tuple[str, ...]
    coverage_mask: tuple[bool, ...]
    rmse_microVolt: float | None
    mae_microVolt: float | None
    mape_percent: float | None
    rmse_denominator: int
    mae_denominator: int
    mape_denominator: int
    mape_zero_actual_excluded_dates: tuple[str, ...]
    mape_zero_policy: str
    alignment_rows: tuple[dict, ...]

    @property
    def score_available(self) -> bool:
        return self.eligible_count > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "assessment_id": self.assessment_id,
            "dataset_id": self.dataset_id,
            "declaration_digest": self.declaration_digest,
            "execution_artifact_digest": self.execution_artifact_digest,
            "alignment_contract_version": self.alignment_contract_version,
            "declared_horizon": self.declared_horizon,
            "eligible_count": self.eligible_count,
            "excluded_count": self.excluded_count,
            "eligible_dates": list(self.eligible_dates),
            "excluded_dates": list(self.excluded_dates),
            "coverage_mask": list(self.coverage_mask),
            "score_available": self.score_available,
            "metrics": {
                "rmse_microVolt": self.rmse_microVolt,
                "mae_microVolt": self.mae_microVolt,
                "mape_percent": self.mape_percent,
            },
            "denominators": {
                "rmse": self.rmse_denominator,
                "mae": self.mae_denominator,
                "mape": self.mape_denominator,
            },
            "mape_zero_policy": self.mape_zero_policy,
            "mape_zero_actual_excluded_dates": list(self.mape_zero_actual_excluded_dates),
            "alignment_rows": [dict(row) for row in self.alignment_rows],
        }


def score_verified_observations(alignment: ObservedAlignmentResult) -> ObservedMetricResult:
    """Score exact C1 eligible pairs; retain all excluded rows and evidence.

    RMSE and MAE use all eligible measured targets. MAPE is percent and excludes
    eligible observations whose actual value is exactly zero. A metric with a
    zero denominator is None, never NaN or a fabricated zero.
    """
    if not isinstance(alignment, ObservedAlignmentResult):
        raise TypeError("alignment must be an ObservedAlignmentResult")
    if alignment.contract_version != ALIGNMENT_CONTRACT_VERSION:
        raise ValueError("unsupported C1 alignment contract")
    if not alignment.rows:
        raise ValueError("alignment must retain a nonempty declared horizon")
    dates = [row.target_date for row in alignment.rows]
    if dates != sorted(set(dates)):
        raise ValueError("alignment dates must be strictly increasing and unique")

    eligible_dates: list[str] = []
    excluded_dates: list[str] = []
    zero_dates: list[str] = []
    errors: list[float] = []
    percentage_errors: list[float] = []
    for row in alignment.rows:
        if not isinstance(row.eligible, bool):
            raise ValueError("eligibility must be boolean")
        if not isinstance(row.forecast_microVolt, (int, float)) or not isfinite(row.forecast_microVolt):
            raise ValueError("nonfinite forecast")
        # Recheck the source evidence; a forged C1 row must not create a score.
        if row.eligible != row.observation_evidence.verified_observed:
            raise ValueError("alignment eligibility disagrees with observation evidence")
        if row.eligible:
            actual = row.observation_evidence.observation.value_microVolt
            if (type(actual) not in (int, float) or not isfinite(actual)
                    or row.observed_microVolt != float(actual)
                    or row.exclusion_reasons):
                raise ValueError("eligible row does not match verified measurement")
            eligible_dates.append(row.target_date)
            error = row.forecast_microVolt - actual
            errors.append(error)
            if actual == 0:
                zero_dates.append(row.target_date)
            else:
                percentage_errors.append(abs(error / actual) * 100.0)
        else:
            if row.observed_microVolt is not None or not row.exclusion_reasons:
                raise ValueError("excluded row contains a scoreable observation")
            excluded_dates.append(row.target_date)

    n = len(errors)
    rmse = sqrt(fsum(e * e for e in errors) / n) if n else None
    mae = fsum(abs(e) for e in errors) / n if n else None
    mape_n = len(percentage_errors)
    mape = fsum(percentage_errors) / mape_n if mape_n else None
    if any(v is not None and not isfinite(v) for v in (rmse, mae, mape)):
        raise ValueError("nonfinite metric result")
    return ObservedMetricResult(
        contract_version=SCORING_CONTRACT_VERSION,
        assessment_id=alignment.assessment_id,
        dataset_id=alignment.dataset_id,
        declaration_digest=alignment.declaration_digest,
        execution_artifact_digest=alignment.execution_artifact_digest,
        alignment_contract_version=alignment.contract_version,
        declared_horizon=len(alignment.rows),
        eligible_count=n,
        excluded_count=len(excluded_dates),
        eligible_dates=tuple(eligible_dates),
        excluded_dates=tuple(excluded_dates),
        coverage_mask=alignment.coverage_mask,
        rmse_microVolt=rmse,
        mae_microVolt=mae,
        mape_percent=mape,
        rmse_denominator=n,
        mae_denominator=n,
        mape_denominator=mape_n,
        mape_zero_actual_excluded_dates=tuple(zero_dates),
        mape_zero_policy=MAPE_ZERO_POLICY,
        alignment_rows=tuple(row.to_dict() for row in alignment.rows),
    )
