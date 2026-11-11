#!/usr/bin/env python3
"""Print a deterministic SHA-256 manifest for a split directory."""

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python reproducibility/make_manifest.py SPLIT_DIR")
    directory = Path(sys.argv[1])
    files = sorted(directory.glob("*.txt"))
    if not files:
        raise SystemExit(f"No split text files found in {directory}")
    manifest = {
        "format": 1,
        "files": {
            path.name: {
                "sha256": sha256(path),
                "ids": sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
            }
            for path in files
        },
    }
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
