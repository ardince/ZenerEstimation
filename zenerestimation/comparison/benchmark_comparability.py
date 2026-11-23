"""M19-C2-A: scientific alignment of independently C1-verified entries.

Alignment is NOT cross-model assessment authorization, metric normalization,
ranking, scientific independence, or authentication. No model is executed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from math import isfinite

from zenerestimation.comparison.benchmark_contract import verify_benchmark_entry
from zenerestimation.assessment.holdout_target_disclosure import _digest


@dataclass(frozen=True)
class VerifiedBenchmarkInput:
    """A C1 entry plus its complete upstream verification context.

    Frozen does not freeze nested mappings; verification occurs on every call.
    """
    entry: Mapping
    declaration: object
    eligibility_snapshot: Mapping
    disclosure_snapshot: Mapping
    scoring_snapshot: Mapping
    evaluation_evidence: Mapping


def _observations(entry: Mapping) -> tuple[tuple[str, ...], tuple[float, ...]]:
    dates = entry["forecast_dates"]
    rows = entry["by_date"]
    horizon = entry["forecast_horizon"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("invalid forecast horizon")
    if type(dates) is not list or type(rows) is not list or len(dates) != horizon or len(rows) != horizon:
        raise ValueError("forecast dates/observations do not match horizon")
    if any(type(d) is not str or not d or date.fromisoformat(d).isoformat() != d for d in dates):
        raise ValueError("invalid canonical forecast date")
    if len(set(dates)) != horizon or dates != sorted(dates):
        raise ValueError("forecast dates must be unique and chronological")
    values = []
    for d, row in zip(dates, rows):
        if type(row) is not dict or row.get("target_date") != d:
            raise ValueError("observation row/date mismatch")
        v = row.get("observed_microVolt")
        if type(v) not in (int, float) or not isfinite(v):
            raise ValueError("nonfinite or nonnumeric measured observation")
        values.append(float(v))
    return tuple(dates), tuple(values)


def assess_scientific_alignment(inputs: Sequence[VerifiedBenchmarkInput]) -> dict:
    """Verify each C1 chain, then check exact shared evaluation population.

    Returns an informational scientific alignment result. It NEVER authorizes
    protected holdout comparison or ranking. C2-B must supply authorization.
    """
    if type(inputs) not in (list, tuple) or len(inputs) < 2:
        raise ValueError("at least two benchmark inputs are required")
    profiles = []
    run_ids = set()
    for index, item in enumerate(inputs):
        if not isinstance(item, VerifiedBenchmarkInput):
            raise TypeError(f"input {index} must be VerifiedBenchmarkInput")
        if not verify_benchmark_entry(
            item.entry, declaration=item.declaration,
            eligibility_snapshot=item.eligibility_snapshot,
            disclosure_snapshot=item.disclosure_snapshot,
            scoring_snapshot=item.scoring_snapshot,
            evaluation_evidence=item.evaluation_evidence,
        ):
            raise ValueError(f"input {index} fails C1 upstream verification")
        entry = item.entry
        run_id = entry["run_id"]
        if run_id in run_ids:
            raise ValueError("duplicate run_id")
        run_ids.add(run_id)
        dates, actuals = _observations(entry)
        profiles.append((entry["dataset_id"], entry["forecast_horizon"], dates, actuals))

    reference = profiles[0]
    checks = (
        ("DATASET_MISMATCH", 0),
        ("HORIZON_MISMATCH", 1),
        ("TIMESTAMP_MISMATCH", 2),
        ("OBSERVATION_MISMATCH", 3),
    )
    reasons = [reason for reason, field in checks if any(p[field] != reference[field] for p in profiles[1:])]
    aligned = not reasons
    return {
        "contract_version": "m19_c2a_scientific_alignment_v1",
        "status": "SCIENTIFICALLY_ALIGNED" if aligned else "INCOMPARABLE",
        "authorization": "not_established",
        "ranking_permitted": False,
        "comparison_permitted": False,
        "run_ids": sorted(run_ids),
        "reasons": reasons,
        "shared_dataset_id": reference[0] if aligned else None,
        "shared_forecast_dates": list(reference[2]) if aligned else None,
        "shared_horizon": reference[1] if aligned else None,
        "observation_digest": _digest({"dates": list(reference[2]), "observed_microVolt": list(reference[3])}) if aligned else None,
    }
