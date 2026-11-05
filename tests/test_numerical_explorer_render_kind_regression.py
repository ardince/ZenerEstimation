"""Regression: UI must pass the user-selected kind to numerical_result."""
import ast
from pathlib import Path


def test_render_passes_defined_kind_not_internal_source_kind():
    source = (Path(__file__).resolve().parents[1] / 'zenerestimation' /
              'visualization' / 'cross_battery_dashboard_numerical_ui.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    render = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'render_numerical_explorer')
    calls = [n for n in ast.walk(render) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'numerical_result']
    assert len(calls) == 1
    kind = next(k.value for k in calls[0].keywords if k.arg == 'kind')
    assert isinstance(kind, ast.Name) and kind.id == 'kind'
