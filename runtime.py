"""Shared WSL TensorFlow/TFF runtime for FL-IDS GPU experiments."""
import os
import sys
import hashlib
import importlib.util
import json
from pathlib import Path

if os.name == 'nt':
    raise RuntimeError('Run through start_notebooks.cmd or run.cmd in WSL Ubuntu-22.04.')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

import tensorflow as tf
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import tensorflow_federated as tff

_report = None


def configure():
    global _report
    if _report is not None:
        return _report
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError('TensorFlow GPU required. Check WSL/CUDA environment.')
    with tf.device('/GPU:0'):
        probe = tf.linalg.matmul(tf.ones([8, 8]), tf.ones([8, 8]))
    if 'GPU:0' not in probe.device or float(tf.reduce_sum(probe)) != 512:
        raise RuntimeError('TensorFlow GPU probe failed.')
    tff.backends.native.set_sync_local_cpp_execution_context(max_concurrent_computation_calls=1)
    @tff.tensorflow.computation
    def worker_probe():
        with tf.device('/GPU:0'):
            # GPU-only op: verifies the installed TFF worker has CUDA support.
            size = tf.raw_ops.CudnnRNNParamsSize(num_layers=1, num_units=4,
                input_size=4, T=tf.float32, S=tf.int32, rnn_mode='lstm')
            return size
    size = worker_probe()
    if int(size[0]) <= 0:
        raise RuntimeError('TFF GPU worker probe failed.')
    package = Path(importlib.util.find_spec('tensorflow_federated').origin).parent
    worker = package / 'data' / 'worker_binary'
    digest = hashlib.sha256(worker.read_bytes()).hexdigest()
    _report = {'python': sys.version, 'tensorflow': tf.__version__, 'tff': tff.__version__,
               'gpu': tf.config.experimental.get_device_details(gpus[0]).get('device_name'),
               'tff_worker_sha256': digest, 'tensorflow_gpu_probe': 'passed',
               'tff_cuda_only_probe': 'passed'}
    print(json.dumps(_report, indent=2), flush=True)
    return _report
