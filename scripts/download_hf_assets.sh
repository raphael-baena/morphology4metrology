#!/usr/bin/env bash
# Download pretrained weights and paper dataset from Hugging Face.
# Requires: huggingface_hub (provides the `hf` CLI)
#   pip install huggingface_hub
# Do NOT install the PyPI package `datasets` — it conflicts with this repo's datasets/ module.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HF_MODEL_REPO="${HF_MODEL_REPO:-RaphaelBfr/morphology4metrology}"
HF_DATASET_REPO="${HF_DATASET_REPO:-RaphaelBfr/morphology4metrology-bnf2813}"
DATA_FOLDER="${DATA_FOLDER:-btv1b84472995}"

WEIGHTS_DIR="${WEIGHTS_DIR:-$PROJECT_ROOT/weights}"
DATASETS_PATH="${DATASETS_PATH:-$HOME/datasets}"
DOWNLOAD_WEIGHTS=true
DOWNLOAD_DATASET=true
WRITE_CONFIG=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Download pretrained checkpoint and/or paper dataset from Hugging Face.

Options:
  --weights-only       Download checkpoint only
  --dataset-only       Download dataset only
  --weights-dir PATH   Local dir for checkpoint.pth (default: \$PROJECT_ROOT/weights)
  --datasets-path PATH Parent dir for dataset folder (default: \$HOME/datasets)
  --write-config       Write datasets/config.json with --datasets-path
  -h, --help           Show this help

Environment variables (override defaults):
  HF_MODEL_REPO, HF_DATASET_REPO, DATA_FOLDER, WEIGHTS_DIR, DATASETS_PATH

Example:
  bash scripts/download_hf_assets.sh
  DATASETS_PATH=/data/datasets bash scripts/download_hf_assets.sh --write-config
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights-only)
      DOWNLOAD_DATASET=false
      shift
      ;;
    --dataset-only)
      DOWNLOAD_WEIGHTS=false
      shift
      ;;
    --weights-dir)
      WEIGHTS_DIR="$2"
      shift 2
      ;;
    --datasets-path)
      DATASETS_PATH="$2"
      shift 2
      ;;
    --write-config)
      WRITE_CONFIG=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: 'hf' CLI not found. Install with: pip install huggingface_hub" >&2
  exit 1
fi

mkdir -p "$WEIGHTS_DIR"
mkdir -p "$DATASETS_PATH"

if $DOWNLOAD_WEIGHTS; then
  echo "==> Downloading checkpoint from $HF_MODEL_REPO"
  hf download "$HF_MODEL_REPO" checkpoint.pth --local-dir "$WEIGHTS_DIR"
  echo "    Saved: $WEIGHTS_DIR/checkpoint.pth"
fi

if $DOWNLOAD_DATASET; then
  echo "==> Downloading dataset from $HF_DATASET_REPO (~2.6 GB, may take a while)"
  hf download "$HF_DATASET_REPO" --repo-type dataset --local-dir "$DATASETS_PATH/$DATA_FOLDER"
  echo "    Saved: $DATASETS_PATH/$DATA_FOLDER"
fi

if $WRITE_CONFIG; then
  CONFIG_FILE="$PROJECT_ROOT/datasets/config.json"
  python3 - <<PY
import json
from pathlib import Path

path = Path("$CONFIG_FILE")
path.write_text(
    json.dumps({"datasets_path": "$DATASETS_PATH"}, indent=2) + "\n",
    encoding="utf-8",
)
print(f"    Wrote {path}")
PY
fi

echo ""
echo "Done."
if $DOWNLOAD_WEIGHTS; then
  echo "  Checkpoint:  $WEIGHTS_DIR/checkpoint.pth"
  echo "  Step 0:      --model_checkpoint_path $WEIGHTS_DIR/checkpoint.pth"
fi
if $DOWNLOAD_DATASET; then
  echo "  Dataset:     $DATASETS_PATH/$DATA_FOLDER"
  echo "  Step 0/1/2:  --data_folder $DATA_FOLDER"
fi
if ! $WRITE_CONFIG; then
  echo ""
  echo "Set datasets/config.json:"
  echo "  {\"datasets_path\": \"$DATASETS_PATH\"}"
  echo "Or re-run with: --write-config"
fi
