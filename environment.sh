#!/usr/bin/env bash
# Use the installed WSL Python, TensorFlow and CUDA libraries.
export PATH="/opt/flids-tf/bin:$PATH"
FLIDS_SITE=$(/opt/flids-tf/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
FLIDS_CUDA_LIBS=$(find "$FLIDS_SITE/nvidia" -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH="$FLIDS_CUDA_LIBS:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$FLIDS_SITE/nvidia/cuda_nvcc"
export PATH="$FLIDS_SITE/nvidia/cuda_nvcc/bin:$PATH"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
