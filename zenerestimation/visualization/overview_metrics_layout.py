"""Present provenance notes and benchmark metrics side by side on Overview.

Only reshapes provided evidence; does not derive or reconstruct metrics.
"""
from __future__ import annotations

import pandas as pd


def overview_with_metrics(notes: pd.DataFrame, metrics: pd.DataFrame, *, optimized: bool) -> pd.DataFrame:
    """Return an Excel-ready rectangular table with notes at left and metrics at right."""
    if list(notes.columns) != ['Property', 'Value']:
        raise ValueError('Overview notes require Property and Value')
    if not isinstance(metrics, pd.DataFrame):
        raise TypeError('Metrics must be a DataFrame')
    if optimized:
        required = ['Model', 'Benchmark external RMSE (µV)', 'Optimized external RMSE (µV)',
                    'Optimized − benchmark RMSE (µV)', 'Benchmark external MAE (µV)',
                    'Optimized external MAE (µV)']
        if not metrics.empty and not set(required).issubset(metrics.columns):
            raise ValueError('Optimized comparison metrics schema mismatch')
        selected = metrics.reindex(columns=required).copy(deep=True)
        selected.columns = ['Model', 'Saved benchmark RMSE (µV)', 'Optimized RMSE (µV)',
                            'Optimized − saved RMSE (µV)', 'Saved benchmark MAE (µV)',
                            'Optimized MAE (µV)']

    else:
        required = ['model', 'rmse', 'mae']
        optional = ['mape', 'rmse_rank']

        if not metrics.empty and not set(required).issubset(metrics.columns):
            raise ValueError(
                'Saved benchmark metrics require model, rmse and mae columns'
            )

        selected = metrics.reindex(
            columns=[*required, *optional]
        ).copy(deep=True)

        selected.columns = ['Model', 'Saved RMSE (µV)', 'Saved MAE (µV)',
                            'Saved MAPE', 'Saved RMSE rank']
    if selected['Model'].notna().duplicated().any() and selected['Model'].dropna().duplicated().any():
        raise ValueError('Duplicate model metrics')
    count = max(len(notes), len(selected))
    left = notes.reset_index(drop=True).reindex(range(count))
    right = selected.reset_index(drop=True).reindex(range(count))
    separator = pd.DataFrame({' ': [''] * count})
    return pd.concat([left, separator, right], axis=1)
