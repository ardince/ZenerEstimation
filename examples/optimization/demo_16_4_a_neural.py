"""16.4-A: development-only, bounded neural optimization demonstration.

Run from repository root. Does not read benchmark targets, change frozen
benchmark evidence, or perform final holdout evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.optimization.evaluator import OptimizationEvaluator
from zenerestimation.optimization.search import GenericOptimizer
from zenerestimation.optimization.validation import ExpandingWindowSplitter
from zenerestimation.optimization.neural_artifact import write_neural_training_artifact
from zenerestimation.optimization.neural_reporting import summarize_neural_candidates
from zenerestimation.optimization.spaces import (
    LSTMParameterSpace, GRUParameterSpace, HybridNeuralParameterSpace,
)
from zenerestimation.forecasting.neural.lstm import LSTMForecaster
from zenerestimation.forecasting.neural.gru import GRUForecaster
from zenerestimation.forecasting.hybrid.linear_trend_lstm import LinearTrendLSTMForecaster
from zenerestimation.forecasting.hybrid.kalman_lstm import KalmanLSTMForecaster

BATTERY = '732B-5610410'
CUTOFF = pd.Timestamp('2023-12-01')
HOLDOUT_START = pd.Timestamp('2024-03-01')
MODELS = ('LSTM', 'GRU', 'LinearTrendLSTM', 'KalmanLSTM')


def development_only(dataset: BatteryDataset) -> BatteryDataset:
    """Physically exclude all external benchmark rows before any optimization."""
    frame = dataset.data.copy(deep=True)
    if not {'ds', 'microVolt'}.issubset(frame.columns):
        raise ValueError('Expected ds and microVolt columns')
    dates = pd.to_datetime(frame['ds'], errors='raise')
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError('Invalid or duplicated dataset dates')
    frame = frame.loc[dates <= CUTOFF].copy()
    if frame.empty or pd.Timestamp(frame['ds'].max()) != CUTOFF:
        raise ValueError('Canonical development cutoff 2023-12-01 is missing')
    if (pd.to_datetime(frame['ds']) >= HOLDOUT_START).any():
        raise AssertionError('Holdout contamination')
    return BatteryDataset(frame.reset_index(drop=True))


def candidate_plan(epochs: int, batch_size: int):
    """Small fixed two-candidate grids, independent of holdout outcomes."""
    common = dict(window=(4, 6), units=(16,), epochs=(epochs,), batch_size=(batch_size,))
    return {
        'LSTM': LSTMParameterSpace(**common).candidates(),
        'GRU': GRUParameterSpace(**common).candidates(),
        'LinearTrendLSTM': HybridNeuralParameterSpace(**common).candidates(),
        'KalmanLSTM': HybridNeuralParameterSpace(**common).candidates(),
    }


def model_factory(name: str, seed: int):
    if name == 'LSTM':
        return lambda **p: LSTMForecaster(**p, seed=seed)
    if name == 'GRU':
        return lambda **p: GRUForecaster(**p, seed=seed)
    if name == 'LinearTrendLSTM':
        return lambda **p: LinearTrendLSTMForecaster(
            window=p['window'],
            lstm_model=LSTMForecaster(**p, seed=seed),
        )
    if name == 'KalmanLSTM':
        return lambda **p: KalmanLSTMForecaster(
            window=p['window'],
            lstm_model=LSTMForecaster(**p, seed=seed),
        )
    raise ValueError(f'Unknown neural model: {name}')


def run(args):
    from zenerestimation.data.temporal import TemporalPreprocessor

    dataset = BatteryDataset.from_processed_csv(
        Path('datasets/processed/732B-5610410.csv'), battery=BATTERY,
    )
    development = development_only(dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plans = candidate_plan(args.epochs, args.batch_size)
    if args.dry_run:
        print('DEVELOPMENT ONLY; no training or holdout evaluation')
        for name in MODELS:
            print(name, plans[name])
        return

    for name in MODELS:
        print(f'Optimizing {name} on development folds only...', flush=True)
        evaluator = OptimizationEvaluator(
            ExpandingWindowSplitter(folds=args.folds, validation_steps=args.validation_steps),
            metric='rmse', preprocessor=TemporalPreprocessor(),
            require_verified_alignment=True,
            record_neural_diagnostics=True,
        )
        optimizer = GenericOptimizer(
            evaluator, model=name, max_candidates=args.max_candidates,
            max_epoch_folds=args.max_epoch_folds,
        )
        result = optimizer.optimize(development, model_factory(name, args.seed), plans[name])
        artifact_path = output / f'{name}_neural_training.json'
        write_neural_training_artifact(result, artifact_path)
        report = summarize_neural_candidates(result)
        report_path = output / f'{name}_learning_report.json'
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + '\n', encoding='utf-8')
        print(f'{name}: internal RMSE={result.best_score:.6f} µV; best={result.best_params}')
        print(f'  {artifact_path}; {report_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=2)
    parser.add_argument('--validation-steps', type=int, default=2)
    parser.add_argument('--max-candidates', type=int, default=2)
    parser.add_argument('--max-epoch-folds', type=int, default=48)
    parser.add_argument('--output', default='results/732B-5610410/neural_demo_16_4_a')
    args = parser.parse_args()
    if any(getattr(args, k) <= 0 for k in ('epochs','batch_size','folds','validation_steps','max_candidates','max_epoch_folds')):
        parser.error('All numeric limits must be positive')
    run(args)


if __name__ == '__main__':
    main()
