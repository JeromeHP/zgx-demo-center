#!/usr/bin/env bash
# Print the PyTorch wheel index URL this machine should install from.
#
# Two demos pinned https://download.pytorch.org/whl/nightly/cu130 as a literal
# in their install.sh. That is a CUDA-only index: on an AMD box pip resolves it
# and installs a torch that cannot see the GPU, silently, and the demo then runs
# on CPU while claiming otherwise -- or fails at import, depending on the wheel.
#
# Usage:  pip install --pre torch torchvision torchaudio --index-url "$(bash scripts/torch_index.sh)"
#         bash scripts/torch_index.sh cpu     # force CPU regardless of hardware
#
# ROCm wheels are NOT nightly: the ROCm nightly channel lags and has broken
# builds more often than it has working ones, so the AMD path takes the stable
# index deliberately.
set -uo pipefail

CPU_INDEX="https://download.pytorch.org/whl/cpu"

if [ "${1:-}" = "cpu" ]; then
  echo "$CPU_INDEX"; exit 0
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${ZGX_HW_PROFILE:-}"
if [ -z "$profile" ]; then
  profile="$(python3 -c 'import sys,json; sys.path.insert(0,"'"$here"'/.."); import zgx_hardware; print(zgx_hardware.detect()["profile"])' 2>/dev/null || echo cpu-only)"
fi

case "$profile" in
  nvidia-cuda) echo "https://download.pytorch.org/whl/nightly/cu130" ;;
  amd-rocm)    echo "https://download.pytorch.org/whl/rocm6.4" ;;
  *)           echo "$CPU_INDEX" ;;
esac
