"""Read-only, exact-date aligned seven-sheet Excel evidence export.

Requires openpyxl for Excel serialization. No training or artifact writes.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import pandas as pd

from zenerestimation.visualization.cross_battery_dashboard_numerical_ui import numerical_result
from zenerestimation.visualization.optimized_neural_numerical_explorer import OPTIONS, numerical_result as optimized_result

BATTERY = '732B-5610410'
SHEETS = ('01 Overview', '02 Historical Fits', '03 Historical Errors',
          '04 External Holdout', '05 Holdout Errors', '06 Operational Forecasts',
           '07 Benchmark vs Optimized')
SAVED_SHEETS = (*SHEETS[:6], '07 Benchmark Metrics')


def _dated(frame, name):
    """Normalize dated wide tables, rejecting duplicates and invalid timestamps."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=['Quarter'])
    frame = frame.copy(deep=True)
    if 'Quarter' not in frame:
        raise ValueError(f'{name}: missing Quarter')
    dates = pd.to_datetime(frame['Quarter'], errors='raise')
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError(f'{name}: missing or duplicate dates')
    frame['Quarter'] = dates.dt.strftime('%Y-%m-%d')
    return frame.sort_values('Quarter').reset_index(drop=True)


def _combine(saved, optimized, models, *, measured=False, operational=False):
    """Outer join on exact quarters; no positional matching or missing-value fabrication."""
    saved = _dated(saved, 'saved')
    optimized = _dated(optimized, 'optimized')
    # The original saved adapter may carry extra evidence columns; use only the
    # explicitly requested models and the measured column.
    columns_saved = ['Quarter'] + (['Measured (µV)'] if measured and 'Measured (µV)' in saved else [])
    columns_saved += [m for m in models if m in saved]
    columns_opt = ['Quarter'] + (['Measured (µV)'] if measured and 'Measured (µV)' in optimized else [])
    columns_opt += [m for m in models if m in optimized]
    s = saved[columns_saved].rename(columns={m: f'{m} · saved' for m in models})
    o = optimized[columns_opt].rename(columns={m: f'{m} · optimized' for m in models})
    if measured:
        s = s.rename(columns={'Measured (µV)': 'Measured saved'})
        o = o.rename(columns={'Measured (µV)': 'Measured optimized'})
    result = s.merge(o, on='Quarter', how='outer', validate='one_to_one', sort=True)
    if measured:
        a = pd.to_numeric(result.get('Measured saved', pd.Series(index=result.index, dtype=float)), errors='coerce')
        b = pd.to_numeric(result.get('Measured optimized', pd.Series(index=result.index, dtype=float)), errors='coerce')
        both = a.notna() & b.notna()
        if (abs(a[both] - b[both]) > 1e-8).any():
            raise ValueError('Saved and optimized measured actuals disagree at the same quarter')
        result.insert(1, 'Measured (µV)', a.combine_first(b))
        result = result.drop(columns=['Measured saved', 'Measured optimized'], errors='ignore')
    order = ['Quarter'] + (['Measured (µV)'] if measured else [])
    for m in models:
        for source in ('saved', 'optimized'):
            col = f'{m} · {source}'
            if col not in result:
                result[col] = float('nan')
            order.append(col)
        # For future predictions this is a difference between predictions, not an error.
        if operational:
            col = f'{m} · optimized − saved'
            result[col] = result[f'{m} · optimized'] - result[f'{m} · saved']
            order.append(col)
    return result[order].sort_values('Quarter').reset_index(drop=True)


def build_comparison_sheets(root, run_id, manifest, holdout, future, historical, saved, models):
    """Return seven source-precision DataFrames; never reconstruct absent evidence."""
    from zenerestimation.visualization.cross_battery_dashboard_numerical_ui import orient_numerical_table
    models = tuple(models)
    if not models or len(models) != len(set(models)):
        raise ValueError('Select one or more unique models')
    from zenerestimation.visualization.cross_battery_comparison import load_comparison
    comparison = load_comparison(str(root))
    def old(kind):
        raw, _ = numerical_result(comparison, root, battery_id=BATTERY, kind=kind, models=models)
        return orient_numerical_table(raw, 'Dates as rows')
    def new(kind):
        table, _ = optimized_result(root, manifest, holdout, future, historical, saved, kind=kind, models=models)
        return table

    history_saved = old('historical_fits')
    history_opt = new(OPTIONS[1]) if historical is not None else pd.DataFrame(columns=['Quarter'])
    errors_saved = old('historical_errors')
    errors_opt = new(OPTIONS[2]) if historical is not None else pd.DataFrame(columns=['Quarter'])
    holdout_saved = old('holdout')
    holdout_opt = new(OPTIONS[3])
    holdout_errors_saved = old('holdout_errors')
    holdout_errors_opt = new(OPTIONS[4])
    operational_saved = old('operational')
    operational_opt = new(OPTIONS[5])
    # Saved historical/holdout wide adapter can label multiple measures per model.
    # Select the precise scientific quantity needed for each sheet.
    def select_measure(frame, suffix):
        if frame.empty or 'Quarter' not in frame:
            return pd.DataFrame(columns=['Quarter'])
        result = frame[['Quarter'] + (['Measured (µV)'] if 'Measured (µV)' in frame else [])].copy()
        for m in models:
            col = m if m in frame else f'{m} · {suffix}'
            if col in frame:
                result[m] = frame[col]
        return result
    history_saved = select_measure(history_saved, 'fitted')
    errors_saved = select_measure(errors_saved, 'fitted − measured')
    holdout_saved = select_measure(holdout_saved, 'prediction')
    holdout_errors_saved = select_measure(holdout_errors_saved, 'prediction − measured')
    # The optimized operational view contains both model and "model · saved".
    operational_opt = operational_opt[['Quarter', *[m for m in models if m in operational_opt]]].copy()
    # Ensure saved operational data comes from the saved source adapter, not an
    # optimized view's saved overlay.
    sheets = {
        SHEETS[1]: _combine(history_saved, history_opt, models, measured=True),
        SHEETS[2]: _combine(errors_saved, errors_opt, models),
        SHEETS[3]: _combine(holdout_saved, holdout_opt, models, measured=True),
        SHEETS[4]: _combine(holdout_errors_saved, holdout_errors_opt, models),
        SHEETS[5]: _combine(operational_saved, operational_opt, models, operational=True),
    }
    overview = [
        ('Property', 'Value'), ('Battery', BATTERY), ('Optimized run', run_id),
        ('Selected models', ', '.join(models)),
        ('Historical fits', 'In-sample diagnostic; not independent validation'),
        ('Historical errors', 'Fitted minus physically measured; missing observations blank'),
        ('External holdout', 'Different saved/optimized provenance; exact-date alignment only'),
        ('Holdout errors', 'Prediction minus measured; no cross-protocol accuracy claim'),
        ('Operational forecasts', 'Unvalidated future predictions; optimized minus saved is NOT an error'),
        ('Date alignment', 'Outer join by exact quarter; no interpolation or positional matching'),
        ('Missing values', 'Blank means unavailable, not zero'),
        ('Precision', 'Unrounded source floating-point values; Excel displays up to 15 significant digits'),
    ]
    comparison_metrics, _ = optimized_result(
        root, manifest, holdout, future, historical, saved,
        kind=OPTIONS[0], models=models,
    )
    required = [
        'Model', 'Benchmark external RMSE (µV)', 'Optimized external RMSE (µV)',
        'Optimized − benchmark RMSE (µV)', 'Benchmark external MAE (µV)',
        'Optimized external MAE (µV)',
    ]
    if not comparison_metrics.empty and not set(required).issubset(comparison_metrics.columns):
        raise ValueError('Optimized comparison metrics schema mismatch')
    metric_table = comparison_metrics.reindex(columns=required).copy(deep=True)
    # Recover missing saved MAE from the exact frozen benchmark source used
    # by the Saved Benchmarks workbook. Never overwrite an existing value.
    from zenerestimation.visualization.cross_battery_dashboard_numerical_ui import orient_numerical_table
    saved_metrics, _ = numerical_result(
        comparison, root, battery_id=BATTERY, kind='metrics', models=models,
    )
    if not saved_metrics.empty:
        if not {'model', 'mae'}.issubset(saved_metrics.columns):
            raise ValueError('Saved benchmark MAE source schema mismatch')
        if saved_metrics['model'].isna().any() or saved_metrics['model'].duplicated().any():
            raise ValueError('Missing or duplicate model in saved benchmark MAE source')
        if not metric_table.empty:
            if metric_table['Model'].isna().any() or metric_table['Model'].duplicated().any():
                raise ValueError('Missing or duplicate model in optimized comparison metrics')
            source_mae = pd.to_numeric(saved_metrics.set_index('model')['mae'], errors='raise')
            incoming = metric_table['Model'].map(source_mae)
            current = pd.to_numeric(metric_table['Benchmark external MAE (µV)'], errors='coerce')
            overlap = current.notna() & incoming.notna()
            if ((current[overlap] - incoming[overlap]).abs() > 1e-9).any():
                raise ValueError('Conflicting saved benchmark MAE across evidence sources')
            metric_table['Benchmark external MAE (µV)'] = current.combine_first(incoming)
    metric_table.columns = [
        'Model', 'Saved benchmark RMSE (µV)', 'Optimized RMSE (µV)',
        'Optimized − saved RMSE (µV)', 'Saved benchmark MAE (µV)',
        'Optimized MAE (µV)',
    ]
    sheets = {
        SHEETS[0]: pd.DataFrame(overview[1:], columns=overview[0]),
        **sheets,
        SHEETS[6]: metric_table,
    }
    return sheets


def excel_bytes(sheets, *, display_precision="Full"):
    """Serialize the seven evidence tables to a formatted in-memory XLSX via openpyxl."""
    from io import BytesIO
    from numbers import Real
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    if tuple(sheets) not in (SHEETS, SAVED_SHEETS):
        raise ValueError('Seven required worksheets must be supplied in order')
    if display_precision != "Full" and (type(display_precision) is not int or display_precision not in (2, 4, 6, 10)):
        raise ValueError("Display precision must be Full, 2, 4, 6, or 10")
    numeric_format = "0.000000000000000" if display_precision == "Full" else "0." + "0" * display_precision
    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, frame in sheets.items():
        if frame.columns.empty:
            raise ValueError(f'Empty worksheet schema: {title}')
        ws = workbook.create_sheet(title)
        ws.append([str(c) for c in frame.columns])
        for record in frame.itertuples(index=False, name=None):
            converted = []
            for value in record:
                if pd.isna(value):
                    converted.append(None)
                elif isinstance(value, Real) and not isinstance(value, bool):
                    converted.append(float(value))
                elif isinstance(value, (pd.Timestamp, datetime)):
                    converted.append(value.to_pydatetime() if isinstance(value, pd.Timestamp) else value)
                else:
                    converted.append(str(value))
            ws.append(converted)
        # Summaries apply only to actual signed-error evidence, never to
        # predictions or operational forecast differences. Excel formulas
        # remain live when users edit or filter source values.
        if title in (SHEETS[2], SHEETS[4]) and ws.max_row > 1:
            from openpyxl.utils import get_column_letter
            from openpyxl.styles import Border, Side
            first_data, last_data = 2, ws.max_row
            summary_start = last_data + 2  # one blank separating row
            labels = ('Count', 'Mean (µV)', 'Std dev (µV; sample)')
            for offset, label in enumerate(labels):
                cell = ws.cell(summary_start + offset, 1, label)
                cell.font = Font(bold=True, color='17324F')
                cell.fill = PatternFill('solid', fgColor='EAF1F7')
            for col in range(2, ws.max_column + 1):
                header = str(ws.cell(1, col).value)
                # Error worksheets consist of signed-error model columns;
                # exclude any measured or other non-error evidence columns.
                if header.startswith('Measured'):
                    continue
                letter = get_column_letter(col)
                source = f'{letter}{first_data}:{letter}{last_data}'
                formulas = (
                    f'=COUNT({source})',
                    f'=IF(COUNT({source})>0,AVERAGE({source}),"")',
                    f'=IF(COUNT({source})>1,STDEV({source}),"")',
                )
                for offset, formula in enumerate(formulas):
                    cell = ws.cell(summary_start + offset, col, formula)
                    cell.fill = PatternFill('solid', fgColor='EAF1F7')
                    cell.number_format = '0' if offset == 0 else numeric_format
            ws.cell(summary_start + 3, 1, 'Std dev uses sample n−1; blank when n < 2.')
            ws.cell(summary_start + 3, 1).font = Font(italic=True, color='667788', size=9)
        for cell in ws[1]:
            cell.fill = PatternFill('solid', fgColor='17324F')
            cell.font = Font(bold=True, color='FFFFFF')
            cell.alignment = Alignment(wrap_text=True, vertical='center')
        ws.row_dimensions[1].height = 30
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = (f'A1:{get_column_letter(ws.max_column)}{last_data}'
                              if title in (SHEETS[2], SHEETS[4]) and ws.max_row > 1
                              else ws.dimensions)
        for col in range(1, ws.max_column + 1):
            letter = get_column_letter(col)
            ws.column_dimensions[letter].width = ((26 if col == 1 else 72) if title == SHEETS[0]
                                                  else (18 if col == 1 else 31) if title in (SHEETS[6], SAVED_SHEETS[6])
                                                  else (17 if col == 1 else 25))
        if title == SHEETS[0]:
            for row in ws.iter_rows(min_row=2, min_col=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = numeric_format
        if title != SHEETS[0]:
            for row in ws.iter_rows(min_row=2, min_col=2):
                for cell in row:
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = ('0' if title == SAVED_SHEETS[6] and 'rank' in str(ws.cell(1, cell.column).value).lower()
                                              else numeric_format)
    stream = BytesIO()
    workbook.save(stream)
    return stream.getvalue()
