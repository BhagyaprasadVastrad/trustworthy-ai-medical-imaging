"""Recalculate common metrics and bootstrap intervals from a saved prediction NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation import evaluate_binary_classifier
from src.statistics import bootstrap_ci


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_file', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/generated/prediction_metrics.csv'))
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()

    data = np.load(args.prediction_file)
    required = {'probabilities', 'labels'}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f'Missing arrays: {sorted(missing)}')
    y = data['labels'].reshape(-1)
    p = data['probabilities'].reshape(-1)
    metrics = evaluate_binary_classifier(y, p)

    functions = {
        'accuracy': lambda a, b: evaluate_binary_classifier(a, b)['accuracy'],
        'precision': lambda a, b: evaluate_binary_classifier(a, b)['precision'],
        'recall': lambda a, b: evaluate_binary_classifier(a, b)['recall'],
        'f1': lambda a, b: evaluate_binary_classifier(a, b)['f1'],
        'auroc': roc_auc_score,
        'auprc': average_precision_score,
        'brier': brier_score_loss,
        'ece': lambda a, b: evaluate_binary_classifier(a, b)['ece'],
    }
    rows = []
    for name, fn in functions.items():
        ci = bootstrap_ci(y, p, fn, n_bootstrap=args.bootstrap, seed=42)
        rows.append({
            'metric': name,
            'estimate': metrics[name],
            'ci_lower': ci['lower'],
            'ci_upper': ci['upper'],
            'bootstrap_samples': ci['n_valid'],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'Wrote: {args.output}')


if __name__ == '__main__':
    main()