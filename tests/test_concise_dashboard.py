from pathlib import Path
import ast


def test_module_is_syntax_valid():
    path = Path(__file__).resolve().parents[1] / 'zenerestimation/visualization/concise_dashboard.py'
    ast.parse(path.read_text(encoding='utf-8'))


def test_installer_is_syntax_valid():
    path = Path(__file__).resolve().parents[1] / 'tools/install_concise_dashboard.py'
    ast.parse(path.read_text(encoding='utf-8'))


def test_comparison_does_not_calculate_misleading_gain():
    from zenerestimation.visualization.concise_dashboard import comparison_rows
    import inspect
    import ast

    source = inspect.getsource(comparison_rows)
    tree = ast.parse(source)

    # Ignore docstrings and comments when examining executable code.
    function = tree.body[0]

    if (
        function.body
        and isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
        and isinstance(function.body[0].value.value, str)
    ):
        function.body.pop(0)

    executable_code = ast.unparse(function).lower()

    assert "gain" not in executable_code
    assert "improvement" not in executable_code