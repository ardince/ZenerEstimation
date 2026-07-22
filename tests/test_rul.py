import numpy as np

from zenerestimation.prognostics import RULAnalyzer


def test_analyze():

    analyzer = RULAnalyzer()

    values = np.linspace(100, 160, 100)

    result = analyzer.analyze(

        current_value=160,

        current_drift=2.0,

        sigma=0.5,

        values=values,

        model="Kalman",

    )

    assert result.model == "Kalman"

    assert result.mean > 0

    assert result.threshold > 0