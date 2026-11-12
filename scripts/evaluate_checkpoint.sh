#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root}"
: "${WEIGHTS:?Set WEIGHTS to the locked checkpoint}"
: "${TEST_LIST:?Set TEST_LIST to the matching official test ID file}"

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-NUDT-SIRST}"
case "$DATASET" in
  NUDT-SIRST) DEFAULT_IMAGE_SIZE=256 ;;
  IRSTD-1K|IRSTD-1k) DEFAULT_IMAGE_SIZE=512 ;;
  IRSTDID-SKY) DEFAULT_IMAGE_SIZE=512 ;;
  *)
    echo "Unsupported DATASET: $DATASET" >&2
    exit 2
    ;;
esac
IMAGE_SIZE="${IMAGE_SIZE:-$DEFAULT_IMAGE_SIZE}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-2727}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/outputs/evaluation}"
if [[ "$RESULTS_DIR" != /* ]]; then
  RESULTS_DIR="$ROOT/$RESULTS_DIR"
fi

mkdir -p "$RESULTS_DIR"
cd "$ROOT"
"$PYTHON" test_pr_audit.py \
  --dataset "$DATASET" \
  --image_size "$IMAGE_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --seed "$SEED" \
  --num_workers "$NUM_WORKERS" \
  --data_path "$DATA_ROOT" \
  --weights "$WEIGHTS" \
  --test_split_file "$TEST_LIST" \
  --per_image_csv "$RESULTS_DIR/per_image.csv"
