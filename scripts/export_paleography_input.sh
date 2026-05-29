#!/usr/bin/env bash
# Export paleography_input after step-1 + finetune (uses conda env ocrdino).
#
# Example:
#   ./scripts/export_paleography_input.sh \
#     logs_reconstruction/IWCP_step_1_32x32 \
#     logs_reconstruction/IWCP_finetune_32x32
#
# Extra flags are passed to export_paleography_input.py, e.g. --organize_only --clean

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <step1_dir> <finetune_dir> [export_paleography_input.py options...]" >&2
  exit 1
fi

STEP1_DIR="$1"
FINETUNE_DIR="$2"
shift 2

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate ocrdino

cd "${PROJECT_ROOT}"
exec python export_paleography_input.py \
  --step1_dir "${STEP1_DIR}" \
  --finetune_dir "${FINETUNE_DIR}" \
  "$@"
