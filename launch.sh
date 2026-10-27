#!/usr/bin/env bash
set -euo pipefail
FLIDS_PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$FLIDS_PROJECT/environment.sh"
cd "$FLIDS_PROJECT"
if [[ ${1:-} == notebooks ]]; then
    exec python -m jupyterlab --no-browser --allow-root --ip=127.0.0.1 --port=8891 notebooks
fi
exec python -u -B run.py "$@"
