"""Benign/Attack metrics with fixed labels, including absent-class support."""
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
    accuracy_score, roc_auc_score, average_precision_score)


def classification_metrics(y, scores, task='binary'):
    if task != 'binary': raise ValueError('Only binary classification is enabled.')
    y = np.asarray(y).reshape(-1).astype(int)
    scores = np.asarray(scores)
    if not len(y) or not np.isfinite(scores).all(): raise ValueError('Empty targets or nonfinite predictions.')
    names = ['Benign', 'Attack']
    labels = np.arange(len(names))
    if scores.ndim not in (1, 2) or (scores.ndim == 2 and scores.shape[1] != 1):
        raise ValueError('Expected one Attack probability per sample.')
    scores = scores.reshape(-1)
    if len(scores) != len(y): raise ValueError('Prediction length mismatch.')
    pred = (scores >= 0.5).astype(int)
    if not np.isin(y, labels).all(): raise ValueError('Unknown target label.')
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    result = {'samples': len(y), 'accuracy': float(accuracy_score(y, pred)),
        'macro_precision': float(p.mean()), 'macro_recall': float(r.mean()),
        'macro_f1': float(f.mean()), 'weighted_f1': float(np.average(f, weights=support)),
        'confusion_matrix': cm.tolist(), 'class_names': names,
        'per_class': {n: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]),
            'support': int(support[i])} for i, n in enumerate(names)}}
    benign = y == 0
    result['far'] = float(np.mean(pred[benign] != 0)) if benign.any() else None
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    result.update(precision=float(p[1]), recall=float(r[1]), f1=float(f[1]),
        tn=tn, fp=fp, fn=fn, tp=tp,
        auc=float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        pr_auc=float(average_precision_score(y, scores)) if (y == 1).any() else None)
    return result


def binary_metrics(y, scores):
    return classification_metrics(y, scores, 'binary')
