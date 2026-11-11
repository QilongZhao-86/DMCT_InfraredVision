#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/sam_spl/sam2_ckpt/sam2_hiera_tiny.pt"
URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"

mkdir -p "$(dirname "$DEST")"
if [[ -e "$DEST" ]]; then
  echo "Checkpoint already exists: $DEST" >&2
  exit 1
fi
curl -fL "$URL" -o "$DEST"
echo "Saved SAM 2.0 Hiera Tiny checkpoint to $DEST"
