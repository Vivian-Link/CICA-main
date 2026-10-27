"""MOSI/MOSEI metrics: all-reference regression; neutral-excluded binary scores."""
import numpy as np


def metrics(y, predictions):
    y, p = np.asarray(y, dtype=float), np.asarray(predictions, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Metrics require nonempty vectors with identical shapes')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Metric inputs must be finite')
    corr = None
    if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0:
        corr = float(np.corrcoef(y, p)[0, 1])
    result = {'n_all': len(y), 'n_nonzero': int(np.sum(y != 0)),
              'mae': float(np.abs(y - p).mean()), 'corr': corr,
              'acc7': float((np.rint(np.clip(y, -3, 3)) == np.rint(np.clip(p, -3, 3))).mean() * 100),
              'acc2': None, 'f1': None}
    keep = y != 0
    truth, pred = y[keep] > 0, p[keep] >= 0
    if keep.any():
        score = 0.0
        for c in [False, True]:
            tp = np.sum((truth == c) & (pred == c))
            fp = np.sum((truth != c) & (pred == c))
            fn = np.sum((truth == c) & (pred != c))
            denominator = 2 * tp + fp + fn
            score += np.mean(truth == c) * (2 * tp / denominator if denominator else 0.0)
        result.update(acc2=float((truth == pred).mean() * 100), f1=float(score * 100))
    return result
