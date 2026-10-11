#!/usr/bin/env bash
# Capture this machine's raw hardware-probe output, verbatim, for use as a test
# fixture.
#
# Run this FIRST on any new box -- especially the Strix Halo one. The AMD
# fixtures in tests/test_hardware_detect.py were written from ROCm's documented
# output format, not from a real gfx1151 machine, and a fixture invented from a
# man page is exactly the kind of thing that passes while testing nothing.
# Replace them with what this prints.
#
#   bash scripts/capture_hw_fixture.sh > tests/fixtures/<hostname>.txt
set -uo pipefail

probe() {
  echo "### \$ $*"
  "$@" 2>&1 || echo "(exit $?)"
  echo
}

echo "# captured $(date -Is) on $(hostname) -- $(uname -srm)"
echo
probe uname -m
probe nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
probe nvidia-ctk --version
probe rocminfo
probe rocm-smi --showproductname --csv
probe rocm-smi --showmeminfo vram
probe hipconfig --version
probe cat /opt/rocm/.info/version
probe head -1 /proc/meminfo
probe ls -la /dev/kfd
probe ls -la /dev/dri
probe id -nG
