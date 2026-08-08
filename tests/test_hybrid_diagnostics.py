import numpy as np

#from examples.dataset_demo import summary
from tests.helpers import make_dataset

from zenerestimation.diagnostics import HybridDiagnostics

from zenerestimation.forecasting.hybrid import (
    KalmanLSTMForecaster,
)


def make_model():

    model = KalmanLSTMForecaster()

    dataset = make_dataset()

    model.fit(dataset)

    return model, dataset


def test_constructor():

    model, _ = make_model()

    diag = HybridDiagnostics(model)

    assert diag.model is model


def test_run():

    model, dataset = make_model()

    diag = HybridDiagnostics(model)

    result = diag.run(dataset)

    assert result is diag


def test_verify_decomposition():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    assert diag.verify_decomposition()


def test_variance_explained():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    explained = diag.variance_explained()

    assert 0.0 <= explained <= 1.0


def test_variances():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    assert diag.trend_variance() >= 0.0

    assert diag.residual_variance() >= 0.0


def test_residual_mean():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    assert abs(diag.residual_mean()) < 10


def test_residual_std():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    assert diag.residual_std() >= 0


def test_residual_rmse():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    assert diag.residual_rmse() >= 0


def test_residual_autocorrelation():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    r = diag.residual_autocorrelation()

    assert -1 <= r <= 1


def test_durbin_watson():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    dw = diag.durbin_watson()

    assert 0 <= dw <= 4


def test_ljung_box():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    lb = diag.ljung_box()

    assert "lb_stat" in lb.columns
    assert "lb_pvalue" in lb.columns


def test_summary_keys():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    summary = diag.summary()

    assert "decomposition_ok" in summary

    assert "trend_variance" in summary

    assert "residual_variance" in summary

    assert "variance_explained" in summary


def test_summary_consistency():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    summary = diag.summary()

    assert summary["decomposition_ok"] is True

    assert summary["trend_variance"] == diag.trend_variance()

    assert summary["residual_variance"] == diag.residual_variance()

    assert summary["variance_explained"] == diag.variance_explained()

    assert summary["max_error"] < 1e-8
    assert summary["rmse_error"] < 1e-8
    assert abs(summary["mean_error"]) < 1e-8


def test_quality_score():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    score = diag.quality_score()

    assert 0.0 <= score <= 100.0


def test_quality_grade():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    assert diag.quality_grade() in {

        "Excellent",
        "Very Good",
        "Good",
        "Fair",
        "Poor",

    }


def test_recommendations():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    rec = diag.recommendations()

    assert isinstance(rec, list)

    assert len(rec) > 0


def test_summary_quality():

    model, dataset = make_model()

    diag = HybridDiagnostics(model).run(dataset)

    summary = diag.summary()

    assert "quality_score" in summary

    assert "quality_grade" in summary

    assert "recommendations" in summary