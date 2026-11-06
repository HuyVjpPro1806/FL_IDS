"""Independent all-Attack CIC pool and three Dirichlet binary clients."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEATURES = ['PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS',
    'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS',
    'DURATION_IN', 'DURATION_OUT', 'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT',
    'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 'SRC_TO_DST_SECOND_BYTES',
    'DST_TO_SRC_SECOND_BYTES', 'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
    'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS', 'SRC_TO_DST_AVG_THROUGHPUT',
    'DST_TO_SRC_AVG_THROUGHPUT', 'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
    'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES',
    'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID',
    'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE']
CLASS_NAMES = ['Benign', 'BruteForce', 'Bot', 'DoS', 'DDoS', 'Infiltration', 'WebAttack']
SPLITS = ('fit', 'validation', 'test')
TASKS = ('binary',)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def config(path=None):
    result = json.loads(Path(path or ROOT/'configs/base.json').read_text(encoding='utf-8-sig'))
    if len(result['clients']) != 3 or len(set(result['clients'])) != 3:
        raise ValueError('This protocol requires three distinct client names.')
    if not isinstance(result['max_pool_rows'], int) or result['max_pool_rows'] < 1:
        raise ValueError('Set a positive integer maximum pool size.')
    if result['sampling'] != 'all_clean_attack_then_benign_without_replacement':
        raise ValueError('Keep all clean Attack rows and fill the remaining capacity with Benign.')
    if result['partition']['method'] != 'attack_family_dirichlet' or result['partition']['alpha'] <= 0:
        raise ValueError('Expected attack-family Dirichlet with positive alpha.')
    return result


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def protocol_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def family(label):
    value = ' '.join(str(label).strip().casefold().split())
    if value == 'benign': return 0
    if value in ('ftp-bruteforce', 'ssh-bruteforce'): return 1
    if value == 'bot': return 2
    if value.startswith('dos attacks-'): return 3
    if value.startswith('ddos attack-') or value.startswith('ddos attacks-'): return 4
    if value in ('infilteration', 'infiltration'): return 5
    if value in ('brute force -web', 'brute force -xss', 'sql injection'): return 6
    raise ValueError(f'Unmapped CIC attack label: {label!r}')
