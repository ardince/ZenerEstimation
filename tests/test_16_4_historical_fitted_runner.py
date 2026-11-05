
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PATH = Path("examples/optimization/run_16_4_optimized_neural_forecasts.py")
spec = importlib.util.spec_from_file_location("historical_runner", PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_extract_fitted_preserves_missing_and_warmup():
    dates = pd.date_range("2020-03-01", periods=6, freq="QS-MAR")
    canonical = pd.DataFrame({"ds": dates, "microVolt": [1.,2.,np.nan,4.,5.,6.],
                              "is_observed": ["True","True","False","True","True","True"]})
    prepared = type("Prepared", (), {"data": pd.DataFrame({"ds":dates})})()
    model = type("Model", (), {"window": 2, "fitted": pd.Series([np.nan,np.nan,3.,4.1,5.2,6.3])})()
    rows = mod.extract_historical_fitted("LSTM",model,prepared,canonical,dates[-1])
    assert len(rows) == 6
    assert rows[0]["is_warmup"] and rows[0]["fitted_microvolt"] is None
    assert rows[2]["was_interpolated"] and rows[2]["measured_microvolt"] is None
    assert rows[2]["error_fitted_minus_measured"] is None
    assert rows[3]["error_fitted_minus_measured"] == pytest.approx(0.1)
    assert all(r["fit_scope"] == "development_in_sample" for r in rows)

def test_extract_fitted_rejects_misalignment():
    dates = pd.date_range("2020-03-01", periods=5, freq="QS-MAR")
    canonical = pd.DataFrame({"ds":dates,"microVolt":range(5),"is_observed":["True"]*5})
    prepared = type("Prepared", (), {"data":pd.DataFrame({"ds":dates[::-1]})})()
    model = type("Model", (), {"window":2,"fitted":pd.Series([np.nan,np.nan,2.,3.,4.])})()
    with pytest.raises(ValueError,match="align"):
        mod.extract_historical_fitted("LSTM",model,prepared,canonical,dates[-1])

def test_extract_fitted_rejects_nonfinite_after_warmup():
    dates = pd.date_range("2020-03-01", periods=5, freq="QS-MAR")
    canonical = pd.DataFrame({"ds":dates,"microVolt":range(5),"is_observed":["True"]*5})
    prepared = type("Prepared", (), {"data":pd.DataFrame({"ds":dates})})()
    model = type("Model", (), {"window":2,"fitted":pd.Series([np.nan,np.nan,2.,np.nan,4.])})()
    with pytest.raises(ValueError,match="missing fitted"):
        mod.extract_historical_fitted("LSTM",model,prepared,canonical,dates[-1])
