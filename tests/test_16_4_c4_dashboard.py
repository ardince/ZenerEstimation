from pathlib import Path
from zenerestimation.visualization import model_performance_comparison as module


def test_c4_does_not_import_model_training():
    source = Path(module.__file__).read_text(encoding='utf-8')
    assert 'model.fit(' not in source
    assert 'model.predict(' not in source
    assert '.write_text(' not in source


def test_c4_uses_validated_existing_sources():
    source = Path(module.__file__).read_text(encoding='utf-8')
    for name in ('load_historical_fit', 'load_external_accuracy_evidence',
                 'load_410_operational_forecasts', 'make_410_future_chart',
                 'verified_410_visuals'):
        assert name in source


def test_c4_comparison_requires_all_models(monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'load_historical_fit', lambda root: ({}, {
        'included_models': [], 'excluded_models': {m:'not available' for m in module.MODELS},
        'models': {}, 'dates': []}))
    monkeypatch.setattr(module, 'load_external_accuracy_evidence', lambda path: {
        'metrics': [{'model':'ARIMA', 'rmse':1, 'mae':1,
                     'first_quarter_absolute_error':1}]})
    import pytest
    with pytest.raises(ValueError, match='exactly one row'):
        module.build_comparison(tmp_path)
