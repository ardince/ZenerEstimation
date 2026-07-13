import pytest

from zenerestimation.forecasting.base import BaseForecastModel


def test_base_model_is_abstract():

    with pytest.raises(TypeError):

        BaseForecastModel()