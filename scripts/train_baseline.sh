#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root}"
: "${SPLIT_DIR:?Set SPLIT_DIR to the dataset train/validation ID files}"

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-NUDT-SIRST}"
case "$DATASET" in
  NUDT-SIRST)
    DEFAULT_BATCH_SIZE=12
    DEFAULT_IMAGE_SIZE=256
    ;;
  IRSTD-1K|IRSTD-1k)
    DEFAULT_BATCH_SIZE=8
    DEFAULT_IMAGE_SIZE=512
    ;;
  IRSTDID-SKY)
    DEFAULT_BATCH_SIZE=8
    DEFAULT_IMAGE_SIZE=512
    ;;
  *)
    echo "Unsupported DATASET: $DATASET" >&2
    exit 2
    ;;
esac
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
IMAGE_SIZE="${IMAGE_SIZE:-$DEFAULT_IMAGE_SIZE}"
EPOCHS="${EPOCHS:-300}"
LR="${LR:-0.01}"
SEED="${SEED:-2727}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_DIR="${SAVE_DIR:-outputs/baseline}"
if [[ "$SAVE_DIR" != /* ]]; then
  SAVE_DIR="$ROOT/$SAVE_DIR"
fi

mkdir -p "$SAVE_DIR"
cd "$ROOT"
"$PYTHON" training.py \
  --dataset "$DATASET" \
  --image_size "$IMAGE_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --epoch "$EPOCHS" \
  --lr "$LR" \
  --seed "$SEED" \
  --num_workers "$NUM_WORKERS" \
  --data_path "$DATA_ROOT" \
  --train_split_file "$SPLIT_DIR/train_ids.txt" \
  --val_split_file "$SPLIT_DIR/val_ids.txt" \
  --config_path "$ROOT/infrared_model/configs/spl_t.yaml" \
  --save_dir "$SAVE_DIR" \
  --gpu "${GPU_ID:-0}"
