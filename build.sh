#!/usr/bin/env bash
# build.sh -- Linux/macOS build (UNTESTED on this release: see docs/02_INSTALL_LINUX.md)
set -euo pipefail
cd "$(dirname "$0")/src"
ARCH="${CUDA_ARCH:--arch=sm_60}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
make CUDA_ARCH="$ARCH" -j "$JOBS" gpumd
make CUDA_ARCH="$ARCH" -j "$JOBS" nep
make CUDA_ARCH="$ARCH" gpumd_replica
echo "built: $(pwd)/gpumd $(pwd)/nep $(pwd)/gpumd_replica"
