"""Binary IDS metrics at fixed threshold 0.5."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def binary_metrics(y, scores):
    y, scores = np.asarray(y).reshape(-1), np.asarray(scores).reshape(-1)
    if not np.isfinite(scores).all():
        raise FloatingPointError('Nonfinite predictions; experiment stopped.')
    labels = scores >= 0.5
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, labels, labels=[0, 1]).ravel()]
    return {'samples': len(y), 'accuracy': (tn+tp)/len(y),
            'precision': float(precision_score(y, labels, zero_division=0)),
            'recall': float(recall_score(y, labels, zero_division=0)),
            'f1': float(f1_score(y, labels, zero_division=0)),
            'far': fp/(tn+fp) if tn+fp else None,
            'auc': float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

