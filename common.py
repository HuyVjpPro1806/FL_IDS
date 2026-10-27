"""Shared base configuration and atomic experiment metadata."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IDENTIFIERS = {'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT'}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def config():
    value = json.loads((ROOT / 'configs/base.json').read_text(encoding='utf-8'))
    expected = dict(hidden_units=[12, 6, 3], activation='relu', output_activation='sigmoid',
                    dropout=0.4, batch_size=2048, local_epochs=3, rounds=10,
                    client_learning_rate=0.001, server_learning_rate=0.05,
                    test_fraction=0.3, deduplicate=False,
                    balance='random_undersample_to_minority_before_split')
    for key, target in expected.items():
        if value[key] != target:
            raise ValueError(f'{key} differs from the reference base settings: {value[key]} != {target}')
    if list(value['datasets']) != ['unsw', 'bot', 'cic']:
        raise ValueError('The configured clients must be UNSW, BoT and CIC, in that order.')
    if value.get('preprocessing') != {'method': 'per_client_log_zscore', 'fit': 'fit_only',
            'log': 'signed_log1p', 'std_ddof': 0, 'clip': [-8.0, 8.0]}:
        raise ValueError('This project requires fit-only per-client Log+Z-score and clipping [-8,8].')
    return value


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def protocol_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
