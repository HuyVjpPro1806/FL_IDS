"""Prepare UNSW/BoT/CIC directly from raw CSV, preserving the frozen row splits."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, config, digest_file, dump, protocol_digest
from feature_preprocessing import POLICY, fit_logz, transform_raw

SPLITS = ('fit', 'validation', 'test')


def source_identity(path):
    stat = Path(path).stat()
    return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def protocol():
    settings = config()
    recipe = json.loads((ROOT/'data_splits/manifest.json').read_text(encoding='utf-8'))
    # Paths are project-relative so moving the project does not invalidate the split.
    cfg = json.loads((ROOT/'configs/tuning.json').read_text(encoding='utf-8'))
    folder = ROOT/cfg['run_directory']/'prepared'
    identity = protocol_digest({'settings': settings, 'recipe': recipe,
        'policy': POLICY, 'code': {name: digest_file(ROOT/name) for name in
                                 ('prepare.py', 'feature_preprocessing.py')}})
    return recipe, folder, identity


def load_recipe(name, info):
    path = ROOT/info['recipe_file']
    if digest_file(path) != info['recipe_sha256']:
        raise ValueError(f'{name}: frozen split recipe changed.')
    with np.load(path) as data:
        recipe = {key: data[key] for key in data.files}
    all_ids = np.concatenate([recipe[f'{s}_row_id'] for s in SPLITS])
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError(f'{name}: row IDs overlap across partitions.')
    if len(np.intersect1d(recipe['fit_input_hash'], recipe['validation_input_hash'])):
        raise ValueError(f'{name}: duplicate input groups leak into validation.')
    return recipe


def recover_raw(name, info, recipe):
    source = ROOT/'dataset'/info['source_relative']
    before = source_identity(source)
    if before['bytes'] != info['source_bytes'] or digest_file(source) != info['source_sha256']:
        raise ValueError(f'{name}: source CSV differs from frozen sampling.')
    ids = np.concatenate([recipe[f'{s}_row_id'] for s in SPLITS])
    labels = np.concatenate([recipe[f'{s}_y'].reshape(-1) for s in SPLITS])
    order = np.argsort(ids)
    sorted_ids = ids[order]
    raw = np.empty((len(ids), 39), dtype=np.float64)
    offset = copied = 0
    tick = time.monotonic()
    for frame in pd.read_csv(source, usecols=info['features']+['Label'], chunksize=200000):
        start, stop = np.searchsorted(sorted_ids, [offset, offset+len(frame)])
        if start < stop:
            take = sorted_ids[start:stop] - offset
            destination = order[start:stop]
            if not np.array_equal(frame['Label'].to_numpy()[take], labels[destination]):
                raise ValueError(f'{name}: raw row labels differ from frozen recipe.')
            raw[destination] = frame[info['features']].to_numpy(dtype=np.float64)[take]
            copied += stop-start
        offset += len(frame)
        if time.monotonic()-tick > 25:
            print(f'{name}: scanned {offset:,}; recovered {copied:,}/{len(ids):,}', flush=True)
            tick = time.monotonic()
    if copied != len(ids) or offset != info['source_rows'] or source_identity(source) != before:
        raise RuntimeError(f'{name}: source changed or selected row recovery incomplete.')
    if not np.isfinite(raw).all():
        raise ValueError(f'{name}: nonfinite raw features.')
    hashes = np.array([hashlib.blake2b(row.tobytes(), digest_size=16).hexdigest() for row in raw])
    expected = np.concatenate([recipe[f'{s}_input_hash'] for s in SPLITS])
    if not np.array_equal(hashes, expected):
        raise ValueError(f'{name}: raw feature hashes differ from frozen rows.')
    return raw, before


def verify_prepared(manifest, folder=None):
    recipe, default_folder, identity = protocol()
    folder = Path(folder or default_folder)
    if manifest['identity'] != identity:
        raise ValueError('Prepared Log+Z-score protocol changed. Use a fresh run_directory.')
    if manifest['policy'] != POLICY or manifest['split_identity'] != recipe['identity']:
        raise ValueError('Prepared policy or split identity changed.')
    for name, info in manifest['datasets'].items():
        source = ROOT/'dataset'/recipe['datasets'][name]['source_relative']
        if source_identity(source) != info['source_metadata']:
            raise ValueError(f'{name}: source metadata changed.')
        load_recipe(name, recipe['datasets'][name])
        for file, checksum in info['files'].items():
            if digest_file(folder/name/file) != checksum:
                raise ValueError(f'{name}: prepared file changed: {file}')
        if (folder/name/'train.bin').stat().st_size != info['fit_rows']*160:
            raise ValueError(f'{name}: invalid train record length.')
    return manifest


def prepare(config_path=None):
    recipe, folder, identity = protocol()
    if config_path is not None:
        cfg = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
        requested = ROOT/cfg['run_directory']/'prepared'
        if requested.resolve() != folder.resolve():
            raise ValueError('Use the same data run_directory as configs/tuning.json.')
    marker = folder/'manifest.json'
    if marker.exists():
        result = json.loads(marker.read_text(encoding='utf-8'))
        verify_prepared(result, folder)
        print('Using verified raw-CSV Log+Z-score data:', folder, flush=True)
        return result
    folder.mkdir(parents=True, exist_ok=True)
    incomplete = folder/'INCOMPLETE.json'
    if incomplete.exists():
        raise RuntimeError('Inspect INCOMPLETE.json before restarting preparation.')
    dump(incomplete, {'status': 'reading raw CSV; fitting client Log+Z-score on fit rows only'})
    result = {'identity': identity, 'split_identity': recipe['identity'], 'policy': POLICY,
              'datasets': {}, 'test_used_to_fit_scaler': False}
    for name, info in recipe['datasets'].items():
        print(f'PREPARE {name}: raw CSV -> per-client Log+Z-score', flush=True)
        selection = load_recipe(name, info)
        raw, metadata = recover_raw(name, info, selection)
        count = len(selection['fit_row_id'])
        statistics = fit_logz(raw[:count], info['features'])
        dest = folder/name
        dest.mkdir(exist_ok=True)
        dump(dest/'scaler.json', statistics)
        cursor = 0
        for split in SPLITS:
            rows = len(selection[f'{split}_row_id'])
            x = transform_raw(raw[cursor:cursor+rows], statistics)
            y = selection[f'{split}_y'].astype('float32')
            if split == 'fit':
                np.column_stack([x, y.reshape(-1)]).astype('<f4').tofile(dest/'train.bin')
            else:
                np.savez(dest/f'{split}.npz', x=x, y=y,
                    **{key: selection[f'{split}_{key}'] for key in ('row_id', 'attack', 'input_hash')})
            cursor += rows
        np.savez(dest/'split_ids.npz', train_row_id=selection['fit_row_id'],
                 validation_row_id=selection['validation_row_id'], test_row_id=selection['test_row_id'])
        files = {file: digest_file(dest/file) for file in
                 ('train.bin', 'validation.npz', 'test.npz', 'split_ids.npz', 'scaler.json')}
        audit = {'source_relative': info['source_relative'], 'source_sha256': info['source_sha256'],
            'source_metadata': metadata, 'features': info['features'],
            'fit_rows': count, 'validation_rows': info['splits']['validation']['rows'],
            'test_rows': info['splits']['test']['rows'],
            **{f'{s}_class_counts': info['splits'][s]['class_counts'] for s in SPLITS},
            'scaler_fit': 'fit_raw_csv_only', 'scaler': statistics,
            'raw_feature_hashes_match_frozen_rows': True,
            'shared_input_groups_train_validation': 0,
            'shared_input_groups_outer_train_test_retained': info['shared_input_groups_outer_train_test_retained'],
            'files': files}
        dump(dest/'audit.json', audit)
        result['datasets'][name] = audit
        del raw
        print(f'COMPLETE {name}: fit {count:,}; validation {audit["validation_rows"]:,}; test {audit["test_rows"]:,}', flush=True)
    dump(marker, result)
    verify_prepared(result, folder)
    incomplete.unlink()
    return result


if __name__ == '__main__':
    prepare()
