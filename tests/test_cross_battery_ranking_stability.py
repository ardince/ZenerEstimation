"""Unit tests for Sprint 15.4.3; synthetic inputs, no repository artifacts needed."""
import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_ranking_stability import analyze_ranking_stability


@pytest.fixture
def ranks():
    return pd.DataFrame({
        "model": ["ARIMA", "Kalman", "LSTM", "GRU", "LinearTrendLSTM", "KalmanLSTM"],
        "rank_410": [3, 4, 5, 6, 2, 1],
        "rank_110": [3, 5, 1, 2, 4, 6],
    })


def test_counts_and_rho(ranks):
    out = analyze_ranking_stability(ranks)
    assert len(out.pairwise) == 15
    assert len(out.leave_one_out) == 6
    assert len(out.top_k) == 3
    assert out.summary.iloc[0]["spearman_rho"] == pytest.approx(-0.7714285714285715)


def test_pairwise_counts(ranks):
    out = analyze_ranking_stability(ranks)
    assert out.summary.iloc[0]["reversed_pairs"] == 12
    assert out.summary.iloc[0]["concordant_pairs"] == 3
    assert out.summary.iloc[0]["reversal_fraction"] == pytest.approx(0.8)


def test_top_k_overlap(ranks):
    out = analyze_ranking_stability(ranks)
    assert out.top_k["overlap_count"].tolist() == [0, 0, 1]
    assert out.top_k.iloc[2]["shared_models"] == "ARIMA"


def test_leave_one_out_is_finite(ranks):
    out = analyze_ranking_stability(ranks)
    assert out.leave_one_out["spearman_rho"].between(-1, 1).all()
    assert (out.leave_one_out["remaining_models"] == 5).all()


def test_order_invariant(ranks):
    a = analyze_ranking_stability(ranks)
    b = analyze_ranking_stability(ranks.sample(frac=1, random_state=4))
    pd.testing.assert_frame_equal(a.pairwise, b.pairwise)
    pd.testing.assert_frame_equal(a.leave_one_out, b.leave_one_out)
    pd.testing.assert_frame_equal(a.top_k, b.top_k)


@pytest.mark.parametrize("mutation", [
    lambda x: x.drop(columns=["rank_110"]),
    lambda x: x.iloc[:-1],
    lambda x: x.assign(rank_410=[1] * 6),
    lambda x: x.assign(rank_110=[1, 2, 3, 4, 5, float("nan")]),
    lambda x: x.assign(model=["ARIMA"] * 6),
])
def test_reject_invalid_input(ranks, mutation):
    with pytest.raises(ValueError):
        analyze_ranking_stability(mutation(ranks))


def test_input_not_modified(ranks):
    before = ranks.copy(deep=True)
    analyze_ranking_stability(ranks)
    pd.testing.assert_frame_equal(ranks, before)
