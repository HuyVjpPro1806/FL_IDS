"""Clean CIC globally; keep every Attack, fill with Benign, then form Dirichlet clients."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, FEATURES, CLASS_NAMES, SPLITS, config, dump, digest_file, protocol_digest, family
from feature_preprocessing import POLICY, fit_logz, transform_raw

CLEANING_POLICY = {
    'duplicate_key': '39_raw_float64_predictors',
    'keep': 'first_source_row',
    'conflict_target': 'binary_label',
    'conflict_action': 'drop_entire_input_group',
    'attack_metadata': 'representative_and_all_source_attack_labels',
}


def training_settings(path=None):
    return json.loads(Path(path or ROOT/'configs/tuning.json').read_text(encoding='utf-8-sig'))


def settings_base(settings):
    return config(ROOT/settings.get('base_config', 'configs/base.json'))


def source_identity(path):
    stat = Path(path).stat()
    return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def mix64(values, salt=0):
    """Deterministic SplitMix64 priorities; independent salts for group and row decisions."""
    with np.errstate(over='ignore'):
        x = np.asarray(values, dtype=np.uint64) + np.uint64(salt) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def global_split(group_hash, base):
    fractions = np.asarray(base['group_pool_fractions'], dtype=float)
    if fractions.shape != (3,) or not np.isclose(fractions.sum(), 1) or (fractions <= 0).any():
        raise ValueError('Invalid group pool fractions.')
    seed = int(base['seed'])
    split_u = (mix64(group_hash, seed) >> np.uint64(11)).astype(float) / 2**53
    split = np.searchsorted(np.cumsum(fractions)[:-1], split_u).astype(np.uint8)
    return split


def apportion(total, weights, capacity=None):
    weights = np.asarray(weights, dtype=float)
    if total < 0 or weights.sum() <= 0: raise ValueError('Invalid allocation.')
    ideal = total*weights/weights.sum()
    counts = np.floor(ideal).astype(int)
    order = np.argsort(-(ideal-counts), kind='stable')
    counts[order[:total-int(counts.sum())]] += 1
    if capacity is None: return counts
    capacity = np.asarray(capacity, dtype=int)
    if capacity.shape != counts.shape or (capacity < 0).any() or capacity.sum() < total:
        raise ValueError('Insufficient source capacity for the requested quota.')
    counts = np.minimum(counts, capacity)
    while counts.sum() < total:
        remaining = total-int(counts.sum())
        eligible = capacity > counts
        extra = apportion(remaining, weights*eligible)
        extra = np.minimum(extra, capacity-counts)
        if not extra.any(): extra[np.flatnonzero(eligible)[0]] = 1
        counts += extra
    return counts


def clean_source_groups(group_hash, families, attack_codes):
    """Remove invalid rows and binary-conflict groups, then keep the first source row.

    Array positions are the original CSV row IDs, including invalid rows. Different
    attack subtypes with the same binary target are metadata, not binary conflicts.
    """
    group_hash = np.asarray(group_hash, dtype=np.uint64)
    families = np.asarray(families, dtype=np.uint8)
    attack_codes = np.asarray(attack_codes, dtype=np.uint8)
    if not (group_hash.ndim == families.ndim == attack_codes.ndim == 1
            and len(group_hash) == len(families) == len(attack_codes)):
        raise ValueError('Source index arrays must be aligned vectors.')
    if not np.isin(families, list(range(len(CLASS_NAMES)))+[255]).all():
        raise ValueError('Unknown source attack family.')
    valid_ids = np.flatnonzero(families < len(CLASS_NAMES))
    if not len(valid_ids): raise ValueError('No finite, labeled source rows remain.')
    if attack_codes[valid_ids].max() >= 64: raise ValueError('Attack metadata supports at most 64 labels.')
    order = valid_ids[np.argsort(group_hash[valid_ids], kind='stable')]
    hashes = group_hash[order]
    starts = np.flatnonzero(np.r_[True, hashes[1:] != hashes[:-1]])
    lengths = np.diff(np.r_[starts, len(order)])
    binary = (families[order] != 0).astype(np.uint8)
    conflicts = np.minimum.reduceat(binary, starts) != np.maximum.reduceat(binary, starts)
    masks = np.bitwise_or.reduceat(np.left_shift(np.uint64(1), attack_codes[order].astype(np.uint64)), starts)
    keep_groups = np.flatnonzero(~conflicts)
    representative_ids = order[starts[keep_groups]]
    by_source = np.argsort(representative_ids)
    keep_groups = keep_groups[by_source]
    ids = representative_ids[by_source].astype(np.int64)
    kept_masks = masks[keep_groups]
    ambiguous = (kept_masks & (kept_masks-np.uint64(1))) != 0
    conflict_rows = int(lengths[conflicts].sum())
    cleaned = {'row_id': ids, 'group_hash': group_hash[ids], 'family': families[ids],
        'attack_code': attack_codes[ids], 'source_attack_mask': kept_masks,
        'source_group_rows': lengths[keep_groups].astype(np.uint32),
        'attack_metadata_ambiguous': ambiguous}
    stats = {'raw_rows': len(group_hash), 'finite_rows': len(order),
        'excluded_nonfinite': len(group_hash)-len(order),
        'binary_conflict_groups_removed': int(conflicts.sum()),
        'binary_conflict_rows_removed': conflict_rows,
        'duplicate_rows_removed': len(order)-conflict_rows-len(ids),
        'clean_rows': len(ids), 'clean_benign': int(np.count_nonzero(families[ids] == 0)),
        'clean_attack': int(np.count_nonzero(families[ids] != 0)),
        'groups_with_multiple_original_attack_labels_retained': int(ambiguous.sum()),
        'representative_family_counts': np.bincount(families[ids], minlength=7).tolist(),
        'representative_attack_counts': np.bincount(attack_codes[ids], minlength=int(attack_codes.max())+1).tolist()}
    return cleaned, stats


def _source_index(base, directory):
    if base.get('cleaning') != CLEANING_POLICY: raise ValueError('Unknown source cleaning policy.')
    source = ROOT/base['source_csv']
    source_state = source_identity(source)
    key = protocol_digest({'source': source_state, 'expected_sha256': base.get('expected_source_sha256'),
        'features': FEATURES, 'cleaning': CLEANING_POLICY, 'pandas': pd.__version__,
        'index_code': digest_file(Path(__file__))})
    marker, archive = directory/'source_index.json', directory/'source_index.npz'
    if marker.exists() and archive.exists():
        metadata = json.loads(marker.read_text(encoding='utf-8'))
        if metadata['identity'] == key:
            with np.load(archive) as z: arrays = {k: z[k] for k in z.files}
            print(f'Reusing cleaned source index: {len(arrays["group_hash"]):,} unique inputs', flush=True)
            return arrays, metadata
    digest = digest_file(source)
    if base.get('expected_source_sha256') and digest != base['expected_source_sha256']:
        raise ValueError('Source CSV checksum differs from the declared dataset.')
    chunks = {'group_hash': [], 'family': [], 'attack_code': []}
    vocabulary = []
    invalid = 0
    offset = 0
    for frame in pd.read_csv(source, usecols=FEATURES+['Label', 'Attack'],
            dtype={f: np.float64 for f in FEATURES}, chunksize=base['chunksize']):
        raw = frame[FEATURES]
        finite = np.isfinite(raw.to_numpy()).all(axis=1)
        invalid += int((~finite).sum())
        names = frame['Attack'].astype(str).str.strip()
        for label in sorted(names.unique()):
            family(label)
            if label not in vocabulary: vocabulary.append(label)
        if len(vocabulary) > 64: raise ValueError('Attack metadata supports at most 64 labels.')
        lookup = {v: i for i, v in enumerate(vocabulary)}
        raw_codes = names.map(lookup).to_numpy(dtype=np.uint8)
        families = names.map({v: family(v) for v in names.unique()}).to_numpy(dtype=np.uint8)
        if not np.array_equal(frame['Label'].to_numpy(), (families != 0).astype(int)):
            raise ValueError('Binary Label disagrees with the Attack mapping.')
        families[~finite] = 255
        chunks['group_hash'].append(pd.util.hash_pandas_object(raw, index=False).to_numpy(dtype=np.uint64))
        chunks['family'].append(families)
        chunks['attack_code'].append(raw_codes)
        offset += len(frame)
        if offset % 1000000 == 0:
            print(f'INDEX {offset:,} raw rows; excluded nonfinite: {invalid:,}', flush=True)
    if source_identity(source) != source_state: raise RuntimeError('Source changed during indexing.')
    arrays = {k: np.concatenate(v) for k, v in chunks.items()}
    del chunks
    print('CLEAN full source: remove binary conflicts, then duplicate inputs', flush=True)
    arrays, cleaning = clean_source_groups(arrays['group_hash'], arrays['family'], arrays['attack_code'])
    if cleaning['excluded_nonfinite'] != invalid: raise RuntimeError('Source cleaning counts disagree.')
    print(f'CLEAN {cleaning["clean_rows"]:,} unique inputs; '
        f'{cleaning["duplicate_rows_removed"]:,} duplicate rows removed; '
        f'{cleaning["binary_conflict_rows_removed"]:,} binary-conflict rows removed', flush=True)
    metadata = {'identity': key, 'source_sha256': digest, 'source_state': source_state,
        'source_rows': offset, 'excluded_nonfinite': invalid, 'attack_vocabulary': vocabulary,
        'class_names': CLASS_NAMES, 'group_hash': 'pandas hash of 39 ordered float64 raw features (uint64)',
        'cleaning_policy': CLEANING_POLICY, 'cleaning': cleaning,
        'attack_metadata_policy': 'Original Attack of first source row; mask retains all source Attack labels.'}
    np.savez_compressed(archive, **arrays)
    dump(marker, metadata)
    return arrays, metadata


def audit_groups(recipes, require_unique=False):
    rows, hashes, cells = [], [], []
    for code, recipe in enumerate(recipes.values()):
        rows.append(recipe['row_id'])
        hashes.append(recipe['group_hash'])
        cells.append(np.full(len(recipe['row_id']), code, dtype=np.uint8))
    rows, hashes, cells = np.concatenate(rows), np.concatenate(hashes), np.concatenate(cells)
    if len(np.unique(rows)) != len(rows): raise ValueError('A source row belongs to multiple partitions.')
    order = np.argsort(hashes, kind='stable')
    same = hashes[order][1:] == hashes[order][:-1]
    leaks = int(np.count_nonzero(same & (cells[order][1:] != cells[order][:-1])))
    if leaks: raise ValueError(f'{leaks} duplicate input groups cross client/split boundaries.')
    if require_unique and same.any(): raise ValueError('Duplicate inputs remain within a partition.')
    return {'selected_rows': len(rows), 'unique_row_ids': len(rows),
        'shared_input_groups_between_any_client_or_split': leaks,
        'unique_selected_input_groups': int(len(hashes)-same.sum()),
        'duplicate_input_rows': int(same.sum())}


def row_digest(ids):
    return hashlib.sha256(np.sort(np.asarray(ids, dtype='<i8')).tobytes()).hexdigest()


def select_pool(arrays, base):
    """Keep all unique binary-positive inputs; deterministically sample only Benign."""
    families = arrays['family']
    if not np.isin(families, np.arange(len(CLASS_NAMES))).all():
        raise ValueError('Pool selection requires globally cleaned source metadata.')
    attack = np.flatnonzero(families != 0)
    benign = np.flatnonzero(families == 0)
    cap = int(base['max_pool_rows'])
    if len(attack) > cap:
        raise ValueError(f'{len(attack):,} clean Attack inputs exceed max_pool_rows={cap:,}; '
                         'increase the cap rather than silently dropping Attack.')
    n_benign = min(len(benign), cap-len(attack))
    if n_benign and n_benign < len(benign):
        priorities = mix64(arrays['row_id'][benign].astype(np.uint64), base['seed']+999983)
        benign = benign[np.argpartition(priorities, n_benign-1)[:n_benign]]
    else: benign = benign[:n_benign]
    ids = np.r_[attack, benign]
    ids = ids[np.argsort(arrays['row_id'][ids])]
    selection = {'method': base['sampling'], 'max_pool_rows': cap,
        'source_clean_rows': len(families), 'source_clean_attack': len(attack),
        'source_clean_benign': int(np.count_nonzero(families == 0)),
        'selected_rows': len(ids), 'selected_attack': len(attack), 'selected_benign': n_benign,
        'attack_fraction': len(attack)/len(ids) if len(ids) else None,
        'all_clean_attack_rows_selected': True,
        'all_clean_attack_row_ids_sha256': row_digest(arrays['row_id'][attack]),
        'selected_original_attack_counts': np.bincount(arrays['attack_code'][ids],
                                                     minlength=int(arrays['attack_code'].max())+1).tolist()}
    return {k: v[ids] for k,v in arrays.items()}, selection


def partition_pool(arrays, base):
    """Share each metadata group across three clients; use the same draw for all splits."""
    if len(np.unique(arrays['row_id'])) != len(arrays['row_id']) or \
            len(np.unique(arrays['group_hash'])) != len(arrays['group_hash']):
        raise ValueError('Pool must contain unique source rows and unique inputs.')
    split = global_split(arrays['group_hash'], base)
    alpha = float(base['partition']['alpha'])
    if not np.isfinite(alpha) or alpha <= 0: raise ValueError('Expected positive finite alpha.')
    seed = int(base['partition']['seed'])
    names = list(base['clients'])
    shares = np.random.default_rng(seed).dirichlet([alpha]*3, size=len(CLASS_NAMES))
    recipes, allocations = {}, {}
    for split_i,part in enumerate(SPLITS):
        members = np.flatnonzero(split == split_i)
        owner = np.full(len(arrays['row_id']), 255, dtype=np.uint8)
        matrix = np.zeros((len(CLASS_NAMES), 3), dtype=int)
        for group in range(len(CLASS_NAMES)):
            candidates = members[arrays['family'][members] == group]
            priorities = mix64(arrays['row_id'][candidates].astype(np.uint64), seed+104729+split_i)
            candidates = candidates[np.argsort(priorities, kind='stable')]
            matrix[group] = apportion(len(candidates), shares[group])
            bounds = np.r_[0, np.cumsum(matrix[group])]
            for i in range(3): owner[candidates[bounds[i]:bounds[i+1]]] = i
        allocations[part] = matrix.tolist()
        for i,name in enumerate(names):
            ids = np.flatnonzero(owner == i)
            ids = ids[np.argsort(arrays['row_id'][ids])]
            if not len(ids): raise ValueError(f'{name}/{part} is empty; explicitly choose another seed.')
            recipes[f'{name}/{part}'] = {'row_id': arrays['row_id'][ids], 'group_hash': arrays['group_hash'][ids],
                'attack_code': arrays['attack_code'][ids], 'attack_family': arrays['family'][ids],
                'source_attack_mask': arrays['source_attack_mask'][ids],
                'source_group_rows': arrays['source_group_rows'][ids],
                'attack_metadata_ambiguous': arrays['attack_metadata_ambiguous'][ids],
                'y_binary': (arrays['family'][ids] != 0).astype(np.int32)}
    recipes = {f'{n}/{s}': recipes[f'{n}/{s}'] for n in names for s in SPLITS}
    if sum(len(v['row_id']) for v in recipes.values()) != len(arrays['row_id']):
        raise ValueError('Every selected pool row must be assigned exactly once.')
    partition = {'method': 'attack_family_dirichlet', 'alpha': alpha, 'seed': seed,
        'metadata_groups': CLASS_NAMES, 'clients': names, 'sampled_group_client_shares': shares.tolist(),
        'allocation_counts': allocations, 'client_size_quota': None, 'attack_fraction_quota': None,
        'split_method': 'source input hash; same global boundary as the prior CIC study',
        'global_split_seed': int(base['seed']),
        'global_split_fractions': base['group_pool_fractions'],
        'global_split_rows': {s: int(np.count_nonzero(split == i)) for i,s in enumerate(SPLITS)},
        'rounding': 'largest remainder within each metadata group and global split'}
    audit_groups(recipes, require_unique=True)
    return recipes, partition


def build_recipes(base, arrays, directory):
    pool, selection = select_pool(arrays, base)
    recipes, partition = partition_pool(pool, base)
    summary, audit_rows = {}, []
    for key,recipe in recipes.items():
        name,part = key.split('/')
        path = directory/name/f'{part}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **recipe)
        counts = np.bincount(recipe['attack_family'], minlength=len(CLASS_NAMES)).tolist()
        n = len(recipe['row_id'])
        summary.setdefault(name,{})[part] = {'rows': n, 'benign': counts[0], 'attack': sum(counts[1:]),
            'attack_fraction': sum(counts[1:])/n, 'class_counts': counts,
            'ambiguous_original_attack_inputs': int(recipe['attack_metadata_ambiguous'].sum()),
            'recipe_file': str(path.relative_to(ROOT)), 'recipe_sha256': digest_file(path)}
        audit_rows.append({'client': name, 'split': part, **summary[name][part], **dict(zip(CLASS_NAMES, counts))})
        print(f'SELECT {key}: {n:,} rows; {sum(counts[1:]):,} attack', flush=True)
    audit = audit_groups(recipes, require_unique=True)
    pd.DataFrame(audit_rows).to_csv(directory/'split_counts.csv', index=False)
    dump(directory/'pool_selection.json', selection)
    dump(directory/'partition.json', partition)
    return recipes, summary, audit, selection, partition


def _recover_raw(base, recipes, folder):
    source_state = source_identity(ROOT/base['source_csv'])
    ids = np.concatenate([v['row_id'] for v in recipes.values()])
    expected_y = np.concatenate([v['attack_family'] for v in recipes.values()])
    expected_hash = np.concatenate([v['group_hash'] for v in recipes.values()])
    order = np.argsort(ids)
    sorted_ids = ids[order]
    path = folder/'selected_raw.npy'
    raw = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(len(ids), 39))
    offset = copied = 0
    for frame in pd.read_csv(ROOT/base['source_csv'], usecols=FEATURES+['Attack'],
            dtype={f: np.float64 for f in FEATURES}, chunksize=base['chunksize']):
        start, stop = np.searchsorted(sorted_ids, [offset, offset+len(frame)])
        if start < stop:
            take = sorted_ids[start:stop]-offset
            dest = order[start:stop]
            values = frame[FEATURES].to_numpy()[take]
            if not np.isfinite(values).all(): raise ValueError('Nonfinite selected source features.')
            actual_hash = pd.util.hash_pandas_object(frame[FEATURES].iloc[take], index=False).to_numpy(dtype=np.uint64)
            if not np.array_equal(actual_hash, expected_hash[dest]): raise ValueError('Source/index feature mismatch.')
            labels = frame['Attack'].iloc[take].map(family).to_numpy()
            if not np.array_equal(labels, expected_y[dest]): raise ValueError('Source/index label mismatch.')
            raw[dest] = values
            copied += stop-start
        offset += len(frame)
        if offset % 1000000 == 0: print(f'READ {offset:,} source rows; recovered {copied:,}', flush=True)
    raw.flush()
    if source_identity(ROOT/base['source_csv']) != source_state: raise RuntimeError('Source changed during recovery.')
    if copied != len(ids): raise RuntimeError('Did not recover every selected source row.')
    return raw, path


def data_identity(settings, base):
    return protocol_digest({'base': base, 'source': source_identity(ROOT/base['source_csv']),
        'policy': POLICY, 'features': FEATURES, 'split_directory': settings.get('split_directory', 'data_splits'),
        'code': {n: digest_file(ROOT/n) for n in ('common.py', 'prepare.py', 'feature_preprocessing.py')}})


def prepare(config_path=None):
    settings = training_settings(config_path)
    base = settings_base(settings)
    folder = ROOT/settings['run_directory']/'prepared'
    directory = ROOT/settings.get('split_directory', 'data_splits')
    identity = data_identity(settings, base)
    marker = folder/'manifest.json'
    if marker.exists():
        manifest = json.loads(marker.read_text(encoding='utf-8'))
        if manifest['identity'] != identity: raise ValueError('Prepared protocol changed; use a new run directory.')
        verify_prepared(manifest, folder, deep=False)
        print('Reusing verified prepared data:', folder, flush=True)
        return manifest
    folder.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    arrays, source = _source_index(base, directory)
    recipes, summary, audit, selection, partition = build_recipes(base, arrays, directory)
    del arrays
    dump(directory/'manifest.json', {'identity': identity, 'source': source, 'clients': summary, 'audit': audit,
                                   'selection': selection, 'partition': partition})
    raw, raw_path = _recover_raw(base, recipes, folder)
    manifest = {'identity': identity, 'source': source, 'policy': POLICY, 'features': FEATURES,
        'class_names': CLASS_NAMES, 'datasets': {}, 'audit': audit, 'test_used_to_fit_scaler': False,
        'tasks': ['binary'], 'cleaning_policy': CLEANING_POLICY, 'selection': selection, 'partition': partition}
    cursor = 0
    for name in base['clients']:
        client = folder/name
        client.mkdir(exist_ok=True)
        n_fit = len(recipes[f'{name}/fit']['row_id'])
        scaler = fit_logz(raw[cursor:cursor+n_fit], FEATURES)
        dump(client/'scaler.json', scaler)
        info = {'scaler': scaler, 'splits': summary[name], 'files': {}}
        for part in SPLITS:
            recipe = recipes[f'{name}/{part}']
            n = len(recipe['row_id'])
            with (client/f'{part}_x.bin').open('wb') as stream:
                for start in range(0, n, 32768):
                    transform_raw(raw[cursor+start:cursor+min(n,start+32768)], scaler).tofile(stream)
            recipe['y_binary'].astype('<i4').tofile(client/f'{part}_y_binary.bin')
            np.savez_compressed(client/f'{part}_meta.npz', **recipe)
            info[part+'_rows'] = n
            for filename in [f'{part}_x.bin', f'{part}_y_binary.bin', f'{part}_meta.npz']:
                info['files'][filename] = {'bytes': (client/filename).stat().st_size,
                    'sha256': digest_file(client/filename)}
            cursor += n
        info['files']['scaler.json'] = {'bytes': (client/'scaler.json').stat().st_size,
            'sha256': digest_file(client/'scaler.json')}
        manifest['datasets'][name] = info
        print(f'PREPARED {name}: fit={n_fit:,}, scaler fit only on these rows', flush=True)
    del raw
    raw_path.unlink()
    dump(marker, manifest)
    verify_prepared(manifest, folder, deep=True)
    return manifest


def verify_prepared(manifest, folder, deep=True):
    recipes = {}
    if manifest.get('cleaning_policy') != CLEANING_POLICY: raise ValueError('Prepared data need global source cleaning.')
    for name, info in manifest['datasets'].items():
        for filename, expected in info['files'].items():
            path = Path(folder)/name/filename
            if path.stat().st_size != expected['bytes'] or (deep and digest_file(path) != expected['sha256']):
                raise ValueError(f'Prepared file differs: {path}')
        for part in SPLITS:
            with np.load(Path(folder)/name/f'{part}_meta.npz') as z:
                recipes[f'{name}/{part}'] = {k: z[k] for k in z.files}
            n = info[part+'_rows']
            recipe = recipes[f'{name}/{part}']
            if any(len(v) != n for v in recipe.values()): raise ValueError('Metadata record count mismatch.')
            if not np.array_equal(recipe['y_binary'], (recipe['attack_family'] != 0).astype(np.int32)):
                raise ValueError('Binary target disagrees with attack metadata.')
            masks = recipe['source_attack_mask']
            representative_bits = np.left_shift(np.uint64(1), recipe['attack_code'].astype(np.uint64))
            if np.any((masks & representative_bits) == 0) or np.any(recipe['source_group_rows'] < 1):
                raise ValueError('Source attack metadata do not include the representative row.')
            if not np.array_equal(recipe['attack_metadata_ambiguous'], (masks & (masks-np.uint64(1))) != 0):
                raise ValueError('Invalid attack ambiguity flags.')
            if (Path(folder)/name/f'{part}_x.bin').stat().st_size != n*39*4:
                raise ValueError('Invalid feature record count.')
            if (Path(folder)/name/f'{part}_y_binary.bin').stat().st_size != n*4:
                raise ValueError('Invalid label record count.')
    if audit_groups(recipes, require_unique=True) != manifest['audit']: raise ValueError('Split audit changed.')
    selection = manifest['selection']
    attack_ids = np.concatenate([v['row_id'][v['y_binary'] == 1] for v in recipes.values()])
    if (len(attack_ids) != selection['source_clean_attack'] or
            row_digest(attack_ids) != selection['all_clean_attack_row_ids_sha256']):
        raise ValueError('Not all cleaned Attack rows were retained exactly once.')
    if (selection['selected_rows'] != manifest['audit']['selected_rows'] or
            selection['selected_rows'] > selection['max_pool_rows'] or
            selection['selected_rows'] != selection['selected_attack']+selection['selected_benign']):
        raise ValueError('Invalid selected pool size.')
    for key,recipe in recipes.items():
        expected = SPLITS.index(key.split('/')[1])
        base = {'seed': manifest['partition']['global_split_seed'],
                'group_pool_fractions': manifest['partition']['global_split_fractions']}
        if not np.all(global_split(recipe['group_hash'], base) == expected):
            raise ValueError('Global train/validation/test input boundary changed.')
    return manifest


def load_partition(folder, name, split, task):
    folder = Path(folder)/name
    x = np.memmap(folder/f'{split}_x.bin', dtype='<f4', mode='r').reshape(-1,39)
    y = np.memmap(folder/f'{split}_y_{task}.bin', dtype='<i4', mode='r')
    if len(x) != len(y): raise ValueError('Feature/label length mismatch.')
    return x, y


if __name__ == '__main__': prepare()
