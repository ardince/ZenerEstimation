import pytest

from zenerestimation.forecasting.neural.training_events import NeuralTrainingRecorder
from zenerestimation.forecasting.neural.lstm import LSTMForecaster
from zenerestimation.forecasting.neural.gru import GRUForecaster


def test_recorder_lifecycle_and_epochs():
    recorder = NeuralTrainingRecorder()
    assert recorder.to_dict()["status"] == "not_started"
    recorder.on_train_begin()
    recorder.on_epoch_begin(0)
    recorder.on_epoch_end(0, {"loss": 0.5, "unused": "text"})
    recorder.on_epoch_begin(1)
    recorder.on_epoch_end(1, {"loss": 0.3})
    recorder.on_train_end()
    trace = recorder.to_dict()
    assert trace["status"] == "completed"
    assert trace["completed_epochs"] == 2
    assert [e["metrics"]["loss"] for e in trace["epochs"]] == [0.5, 0.3]
    assert trace["training_started_at"] is not None
    assert trace["duration_seconds"] >= 0
    assert all(e["duration_seconds"] >= 0 for e in trace["epochs"])


def test_recorder_failure_and_reset():
    recorder = NeuralTrainingRecorder()
    recorder.on_train_begin()
    recorder.mark_failed()
    assert recorder.to_dict()["status"] == "failed"
    recorder.on_train_begin()
    assert recorder.to_dict()["completed_epochs"] == 0
    assert recorder.to_dict()["status"] == "running"


@pytest.mark.parametrize("cls", [LSTMForecaster, GRUForecaster])
def test_opt_in_constructor(cls):
    assert cls().record_training is False
    assert cls(record_training=True).record_training is True
    with pytest.raises(TypeError, match="record_training"):
        cls(record_training=1)


@pytest.mark.parametrize("cls", [LSTMForecaster, GRUForecaster])
def test_opt_in_fit_callback_and_legacy_fit(monkeypatch, cls):
    class DummyHistory:
        history = {"loss": [0.5]}
    class DummyKeras:
        def fit(self, X, y, **kwargs):
            callback = (kwargs.get("callbacks") or [None])[0]
            if callback is not None:
                callback.on_train_begin()
                callback.on_epoch_begin(0)
                callback.on_epoch_end(0, {"loss": 0.5})
                callback.on_train_end()
            self.kwargs = kwargs
            return DummyHistory()
        def predict(self, X, **kwargs):
            import numpy as np
            return np.zeros((len(X), 1))
    class DummyDataset:
        target = list(range(10))
        class Data:
            index = list(range(10))
        data = Data()
    import numpy as np
    for enabled in [False, True]:
        forecaster = cls(window=2, epochs=1, record_training=enabled)
        keras_model = DummyKeras()

        def fake_prepare_data(dataset):
            forecaster.dataset = dataset

            return (
                np.zeros((8, 2, 1)),
                np.zeros(8),
            )

        monkeypatch.setattr(
            forecaster,
            "prepare_data",
            fake_prepare_data,
        )

        monkeypatch.setattr(forecaster, "build_model", lambda: keras_model)
        monkeypatch.setattr(forecaster, "compile_model", lambda: None)
        monkeypatch.setattr(forecaster.scaler, "inverse_transform", lambda values: values)
        forecaster.fit(DummyDataset())
        assert ("callbacks" in keras_model.kwargs) is enabled
        assert (forecaster.training_recorder is not None) is enabled
        if enabled:
            assert forecaster.training_recorder.to_dict()["completed_epochs"] == 1
