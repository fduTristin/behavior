#!/usr/bin/env bash

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

runtime_root="${BEHAVIOR_GRAPHICS_RUNTIME:-/run/ti/BEHAVIOR2026/infra/nv580/runtime}"
driver_version="${BEHAVIOR_GRAPHICS_DRIVER_VERSION:-580.95.05}"
driver_lib_dir="${runtime_root}/usr/lib/x86_64-linux-gnu"
graphics_lib_dir="${runtime_root}/graphics-lib"
support_lib_dir="${runtime_root}/support-lib"
icd_file="${driver_lib_dir}/nvidia_icd.json"

for required_path in "${graphics_lib_dir}/libGLX_nvidia.so.0" "${support_lib_dir}/libwayland-client.so.0" "${icd_file}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing NVIDIA graphics runtime file: ${required_path}" >&2
        exit 1
    fi
done

if [[ "${BEHAVIOR_GRAPHICS_SKIP_DRIVER_CHECK:-0}" != "1" ]]; then
    actual_driver="$({ nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits || true; } | head -n 1 | tr -d '[:space:]')"
    if [[ "${actual_driver}" != "${driver_version}" ]]; then
        echo "NVIDIA userspace/kernel driver mismatch: expected ${driver_version}, got ${actual_driver:-unknown}" >&2
        exit 1
    fi
fi

export LD_LIBRARY_PATH="${graphics_lib_dir}:${support_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VK_ICD_FILENAMES="${icd_file}"

exec "$@"
