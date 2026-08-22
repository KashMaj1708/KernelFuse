#!/usr/bin/env bash
# Source before running kernelfuse / torch extensions on Linux.
set -euo pipefail
TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
