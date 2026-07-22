import numpy as np

from zenerestimation.prognostics import ThresholdEstimator


def test_fixed():

    assert ThresholdEstimator.fixed(35.0) == 35.0


def test_percentile():

    values = np.array([1, 2, 3, 4, 5])

    assert ThresholdEstimator.percentile(values, 100) == 5.0


def test_maximum_margin():

    values = np.array([10, 20, 30])

    assert ThresholdEstimator.maximum_margin(values, 5) == 35.0