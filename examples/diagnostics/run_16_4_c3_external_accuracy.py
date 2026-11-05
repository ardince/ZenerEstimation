"""Create C3 external accuracy evidence using saved files only (no fitting)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zenerestimation.visualization.external_accuracy_evidence import (  # noqa: E402
    BATTERY, build_external_accuracy_evidence, write_external_accuracy_evidence,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'results' / BATTERY / 'external_accuracy_16_4_c3'
    if args.dry_run:
        print('Read-only inputs: optimized_benchmark and historical_fit_diagnostics')
        print('Additive output:', output)
        print('No training, optimization, or benchmark modification')
        return
    evidence = build_external_accuracy_evidence(ROOT)
    paths = write_external_accuracy_evidence(evidence, output)
    print('External holdout: 6 dates; models: 6')
    print('Prediction provenance:', evidence['prediction_source_kind'])
    print('Maximum absolute metric delta:', evidence['max_abs_metric_delta'])
    for item in sorted(evidence['metrics'], key=lambda x: x['rmse']):
        print(f"{item['model']:<18} RMSE={item['rmse']:.6f}  MAE={item['mae']:.6f}  first_abs={item['first_quarter_absolute_error']:.6f}")
    for path in paths:
        print('Saved:', path)


if __name__ == '__main__':
    main()
