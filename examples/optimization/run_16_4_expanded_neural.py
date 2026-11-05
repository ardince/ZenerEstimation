"""Expanded development-only neural search. Run from repository root.

Does not read external benchmark measurements, fit final models, or score holdout.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path

DEMO = Path(__file__).with_name('demo_16_4_a_neural.py')
spec = importlib.util.spec_from_file_location('zener_neural_demo_16_4_a', DEMO)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def expanded_plan(epochs, batch_sizes):
    from zenerestimation.optimization.spaces import LSTMParameterSpace, GRUParameterSpace, HybridNeuralParameterSpace
    common = dict(window=(4,6,8,12),units=(16,32),epochs=tuple(epochs),batch_size=tuple(batch_sizes))
    return {'LSTM':LSTMParameterSpace(**common).candidates(),
            'GRU':GRUParameterSpace(**common).candidates(),
            'LinearTrendLSTM':HybridNeuralParameterSpace(**common).candidates(),
            'KalmanLSTM':HybridNeuralParameterSpace(**common).candidates()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--epochs-grid',type=int,nargs='+',default=[24,48])
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[8,16])
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--validation-steps',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-candidates',type=int,default=32)
    p.add_argument('--max-epoch-folds',type=int,default=3456)
    p.add_argument('--output',default='results/732B-5610410/neural_expanded_16_4')
    args=p.parse_args()
    if any(v<=0 for v in [*args.epochs_grid,*args.batch_sizes,args.folds,args.validation_steps,args.max_candidates,args.max_epoch_folds]):
        p.error('All numeric parameters must be positive')
    plans=expanded_plan(args.epochs_grid,args.batch_sizes)
    print('Expanded candidates per model:',{k:len(v) for k,v in plans.items()})
    print('Planned epoch-fold budget per model:',sum(e*args.folds*4*2*len(args.batch_sizes) for e in args.epochs_grid))
    if args.dry_run:
        print('DRY RUN: no training or holdout access')
        return
    # Reuse the validated 16.4-A optimization and artifact-writing code; override only grid.
    demo.candidate_plan=lambda epochs,batch_size:plans
    args.epochs=max(args.epochs_grid)
    args.batch_size=args.batch_sizes[0]
    demo.run(args)

if __name__=='__main__':main()
