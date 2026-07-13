from zenerestimation.forecasting.base import BaseForecastModel


class DummyForecastModel(BaseForecastModel):

    def fit(self, dataset):

        self.dataset = dataset
        self.is_fitted = True

    def predict(self, steps=1):

        return [0.0] * steps