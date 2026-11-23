"""M19-C1: verified, deterministic, read-only benchmark entry ingestion.

A benchmark entry is not itself a comparability or ranking decision. Upstream
B1--B4 records must be supplied from the authorized assessment workflow.
Digests demonstrate internal consistency, not independent authentication.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping

from zenerestimation.assessment.holdout_evaluation_evidence import (
    verify_holdout_evaluation_evidence,
)
from zenerestimation.assessment.holdout_target_disclosure import _canonical, _digest

BENCHMARK_ENTRY_VERSION = "m19_c1_verified_benchmark_entry_v1"
MODEL_FAMILIES = frozenset({"LSTM", "GRU", "ARIMA", "Kalman", "Hybrid"})
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z", re.ASCII)


def _family_matches(family: str, selected: str) -> bool:
    """Conservative classification: ambiguous names fail closed."""
    normalized = re.sub(r"[\s_-]+", "", selected).lower()
    singles = {"LSTM": "lstm", "GRU": "gru", "ARIMA": "arima", "Kalman": "kalman"}
    if family in singles:
        return normalized == singles[family]
    # Hybrid names must explicitly identify a combination, not merely a base model.
    if normalized in {"lineartrendlstm", "kalmanlstm", "lineartrendgru", "kalmangru"}:
        return True
    if normalized.startswith("hybrid") and len(normalized) > len("hybrid"):
        return True
    tokens = re.findall(r"[a-z]+", selected.lower())
    return len(set(tokens) & {"lstm", "gru", "arima", "kalman"}) >= 2


def _canonical_detach(value: dict, label: str) -> dict:
    if type(value) is not dict:
        raise TypeError(f"{label} must be a plain dict")
    try:
        result = json.loads(_canonical(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if result != value:
        raise ValueError(f"{label} contains noncanonical values")
    return result


def build_benchmark_entry(
    *, run_id: str, model_family: str, declaration,
    eligibility_snapshot: Mapping, disclosure_snapshot: Mapping,
    scoring_snapshot: Mapping, evaluation_evidence: Mapping,
) -> dict:
    """Ingest one fully reverified B4 chain; do not compare or rank entries."""
    if type(run_id) is not str or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id must be a nonempty canonical identifier (max 128 chars)")
    if type(model_family) is not str or model_family not in MODEL_FAMILIES:
        raise ValueError("unsupported model_family")
    if not verify_holdout_evaluation_evidence(
        evaluation_evidence, declaration=declaration,
        eligibility_snapshot=eligibility_snapshot,
        disclosure_snapshot=disclosure_snapshot,
        scoring_snapshot=scoring_snapshot,
    ):
        raise ValueError("B4 evaluation evidence does not verify against B1-B3")
    evidence = _canonical_detach(evaluation_evidence, "B4 evidence")
    selected = evidence["selected_model"]
    if type(selected) is not str or not _family_matches(model_family, selected):
        raise ValueError("model_family does not match verified selected_model")
    payload = {
        "benchmark_entry_version": BENCHMARK_ENTRY_VERSION,
        "status": "VERIFIED_FOR_BENCHMARK_INGESTION",
        "run_id": run_id,
        "model_family": model_family,
        "selected_model": selected,
        "dataset_id": evidence["dataset_id"],
        "assessment_id": evidence["assessment_id"],
        "declaration_digest": evidence["declaration_digest"],
        "development_cutoff": evidence["development_cutoff"],
        "decision_digest": evidence["decision_digest"],
        "plan_digest": evidence["plan_digest"],
        "execution_artifact_digest": evidence["execution_artifact_digest"],
        "training_fingerprint": evidence["training_fingerprint"],
        "eligibility_digest": evidence["eligibility_digest"],
        "disclosure_digest": evidence["disclosure_digest"],
        "scoring_digest": evidence["scoring_digest"],
        "evaluation_evidence_digest": evidence["evaluation_evidence_digest"],
        "forecast_dates": evidence["forecast_dates"],
        "forecast_horizon": evidence["forecast_horizon"],
        "metric_source": "pooled_holdout",
        "error_convention": evidence["error_convention"],
        "metrics": evidence["metrics"],
        "denominators": evidence["denominators"],
        "mape_zero_policy": evidence["mape_zero_policy"],
        "mape_zero_actual_excluded_dates": evidence["mape_zero_actual_excluded_dates"],
        "by_date": evidence["by_date"],
        "scientific_independence": evidence["scientific_independence"],
        "authentication": evidence["authentication"],
    }
    result = _canonical_detach(payload, "benchmark entry payload")
    result["benchmark_entry_digest"] = _digest(result)
    return result


def verify_benchmark_entry(
    entry: Mapping, *, declaration, eligibility_snapshot: Mapping,
    disclosure_snapshot: Mapping, scoring_snapshot: Mapping,
    evaluation_evidence: Mapping,
) -> bool:
    """Fail closed against the complete upstream chain and exact entry schema."""
    if type(entry) is not dict:
        return False
    try:
        expected = build_benchmark_entry(
            run_id=entry["run_id"], model_family=entry["model_family"],
            declaration=declaration, eligibility_snapshot=eligibility_snapshot,
            disclosure_snapshot=disclosure_snapshot, scoring_snapshot=scoring_snapshot,
            evaluation_evidence=evaluation_evidence,
        )
        return _canonical(entry) == _canonical(expected)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
