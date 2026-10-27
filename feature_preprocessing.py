"""Per-client signed log1p -> fit-only population Z-score -> clip [-8, 8]."""
import numpy as np

POLICY = {
    'method': 'per_client_log_zscore',
    'log': 'sign(x) * ln(1 + abs(x)); ln(1+x) for nonnegative features',
    'fit_partition': 'fit only, independently for each client and feature',
    'std_ddof': 0,
    'constant_std_threshold': 1e-12,
    'constant_std_replacement': 1.0,
    'clip': [-8.0, 8.0],
    'output_dtype': 'float32',
    'raw_input': 'original CSV float64 values, without MinMax reconstruction',
}


def log_features(raw):
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 39 or not np.isfinite(values).all():
        raise ValueError('Expected finite raw inputs with shape (n,39).')
    return np.sign(values) * np.log1p(np.abs(values))


def fit_logz(raw, features):
    """Call with fit rows only; never update these statistics at inference."""
    if len(raw) == 0 or len(features) != 39:
        raise ValueError('Expected nonempty fit rows and 39 feature names.')
    total = np.zeros(39, dtype=np.float64)
    squares = np.zeros(39, dtype=np.float64)
    for start in range(0, len(raw), 32768):
        values = log_features(raw[start:start+32768])
        total += values.sum(axis=0)
        squares += np.square(values).sum(axis=0)
    mean = total / len(raw)
    std = np.sqrt(np.maximum(squares / len(raw) - mean * mean, 0.0))
    scale = np.where(std < POLICY['constant_std_threshold'], 1.0, std)
    return {'method': POLICY['method'], 'features': list(features), 'n_fit': len(raw),
            'mean': mean.tolist(), 'std': std.tolist(), 'scale': scale.tolist(),
            'policy': POLICY}


def transform_raw(raw, statistics):
    """Transform raw feature values using an already fitted client scaler."""
    result = (log_features(raw) - np.asarray(statistics['mean'])) / np.asarray(statistics['scale'])
    result = np.clip(result, *POLICY['clip']).astype('<f4')
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite Log+Z-score output.')
    return result
