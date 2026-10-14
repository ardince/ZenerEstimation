"""Sprint 15.4.3: deterministic, read-only cross-battery ranking sensitivity.

This module operates on the already-validated ``ranking_comparison`` DataFrame
from Sprint 15.4.2. It does not load models, write artifacts, or change metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_MODELS = frozenset({
    "ARIMA", "Kalman", "LSTM", "GRU", "LinearTrendLSTM", "KalmanLSTM"
})


@dataclass(frozen=True)
class RankingStabilityResult:
    pairwise: pd.DataFrame
    leave_one_out: pd.DataFrame
    top_k: pd.DataFrame
    summary: pd.DataFrame
    interpretation: str


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "rank_410", "rank_110"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError(f"ranking_comparison requires columns {sorted(required)}")
    x = frame.loc[:, ["model", "rank_410", "rank_110"]].copy()
    if len(x) != 6 or x["model"].isna().any() or set(x["model"]) != EXPECTED_MODELS:
        raise ValueError("Expected exactly the six distinct benchmark models")
    if x["model"].duplicated().any():
        raise ValueError("Duplicate models are not allowed")
    for col in ("rank_410", "rank_110"):
        vals = pd.to_numeric(x[col], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals.to_numpy(dtype=float)).all():
            raise ValueError(f"{col} must contain finite ranks")
        if set(vals.to_numpy(dtype=float)) != set(range(1, 7)):
            raise ValueError(f"{col} must be a permutation of ranks 1..6")
        x[col] = vals.astype(int)
    return x.sort_values("model", kind="stable").reset_index(drop=True)


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of rank arrays (Spearman without ties)."""
    return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])


def analyze_ranking_stability(ranking_comparison: pd.DataFrame) -> RankingStabilityResult:
    """Compute descriptive sensitivity of the six-model cross-battery ranking.

    Pairwise reversals, leave-one-model-out rho, and top-k overlap are all
    deterministic; no resampling, significance testing, or model fitting occurs.
    """
    x = _validate(ranking_comparison)
    indexed = x.set_index("model")
    pair_rows = []
    for left, right in combinations(sorted(EXPECTED_MODELS), 2):
        d410 = int(indexed.at[left, "rank_410"] - indexed.at[right, "rank_410"])
        d110 = int(indexed.at[left, "rank_110"] - indexed.at[right, "rank_110"])
        pair_rows.append({"model_a": left, "model_b": right,
                          "order_410": "a_before_b" if d410 < 0 else "b_before_a",
                          "order_110": "a_before_b" if d110 < 0 else "b_before_a",
                          "reversed": bool(d410 * d110 < 0)})
    pairwise = pd.DataFrame(pair_rows)

    loo_rows = []
    for excluded in sorted(EXPECTED_MODELS):
        remaining = x.loc[x["model"] != excluded]
        loo_rows.append({"excluded_model": excluded, "remaining_models": 5,
                         "spearman_rho": _rho(remaining["rank_410"].to_numpy(),
                                              remaining["rank_110"].to_numpy())})
    leave_one_out = pd.DataFrame(loo_rows)

    top_rows = []
    for k in (1, 2, 3):
        a = set(x.loc[x["rank_410"] <= k, "model"])
        b = set(x.loc[x["rank_110"] <= k, "model"])
        intersection = sorted(a & b)
        top_rows.append({"k": k, "overlap_count": len(intersection),
                         "overlap_fraction": len(intersection) / k,
                         "shared_models": ", ".join(intersection),
                         "top_410": ", ".join(sorted(a)),
                         "top_110": ", ".join(sorted(b))})
    top_k = pd.DataFrame(top_rows)

    reversed_count = int(pairwise["reversed"].sum())
    full_rho = _rho(x["rank_410"].to_numpy(), x["rank_110"].to_numpy())
    summary = pd.DataFrame([{
        "model_count": 6, "pair_count": len(pairwise),
        "concordant_pairs": len(pairwise) - reversed_count,
        "reversed_pairs": reversed_count,
        "reversal_fraction": reversed_count / len(pairwise),
        "spearman_rho": full_rho,
        "loo_rho_min": float(leave_one_out["spearman_rho"].min()),
        "loo_rho_max": float(leave_one_out["spearman_rho"].max()),
    }])
    interpretation = (
        "Descriptive ranking sensitivity only. Battery 410 uses an optimized "
        "benchmark; battery 110 uses transferred configurations. Six models "
        "and two different holdout periods do not support causal transfer "
        "claims, statistical confidence intervals, or population inference. "
        "Saved benchmark rankings remain authoritative."
    )
    return RankingStabilityResult(pairwise, leave_one_out, top_k, summary, interpretation)


def load_ranking_stability(repository_root: str = ".") -> RankingStabilityResult:
    """Load Sprint 15.4.2 validated rankings and analyze them without writes."""
    from zenerestimation.visualization.cross_battery_generalization import load_generalization
    return analyze_ranking_stability(load_generalization(repository_root).ranking_comparison)
