"""Write portable provenance records for training and locked evaluation runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def _source_files(config_paths: list[str | None]) -> list[Path]:
    candidates = [
        REPO_ROOT / "training.py",
        REPO_ROOT / "train_final.py",
        REPO_ROOT / "test_pr_audit.py",
        REPO_ROOT / "metrics.py",
        REPO_ROOT / "scripts" / "train_ours.sh",
        REPO_ROOT / "scripts" / "train_baseline.sh",
        REPO_ROOT / "scripts" / "evaluate_checkpoint.sh",
    ]
    for package in ("training", "dataset", "infrared_model", "metrics"):
        package_root = REPO_ROOT / package
        if package_root.is_dir():
            candidates.extend(package_root.rglob("*.py"))
    candidates.extend(Path(value).expanduser() for value in config_paths if value)
    return sorted({path.resolve() for path in candidates if path.is_file()})


def _copy_split(source_value: str | None, destination: Path) -> dict[str, Any] | None:
    if not source_value:
        return None
    source = Path(source_value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Split file does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) != source_hash:
            raise FileExistsError(
                f"Refusing to overwrite a different archived split: {destination}"
            )
    else:
        shutil.copyfile(source, destination)
    return {
        "archived_file": str(
            destination.relative_to(REPO_ROOT)
            if destination.is_relative_to(REPO_ROOT)
            else destination
        ),
        "sha256": source_hash,
        "id_count": sum(1 for line in source.read_text(encoding="utf-8").splitlines() if line.strip()),
    }


def _read_id_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_training_splits(args: Any) -> dict[str, Any]:
    train_path = Path(args.train_split_file).expanduser().resolve()
    val_path = Path(args.val_split_file).expanduser().resolve()
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError("Both explicit training and internal-validation ID files are required.")

    train_ids = _read_id_list(train_path)
    val_ids = _read_id_list(val_path)
    for label, values in (("train", train_ids), ("validation", val_ids)):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate IDs in {label} split: {label} list must be unique.")
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(f"Training and validation splits overlap ({len(overlap)} IDs).")

    canonical = {
        "NUDT-SIRST": "NUDT-SIRST",
        "IRSTD-1K": "IRSTD-1k",
        "IRSTD-1k": "IRSTD-1k",
        "IRSTDID-SKY": "IRSTDID-SKY",
    }[args.dataset]
    config_dir = Path(args.data_path).expanduser().resolve() / "set_configs" / canonical
    official_train = config_dir / "train.txt"
    official_test = config_dir / "test.txt"
    repository_protocol = REPO_ROOT / "reproducibility" / args.dataset
    if args.dataset == "IRSTDID-SKY":
        repo_train = repository_protocol / "official_train_ids.txt"
        repo_test = repository_protocol / "official_test_ids.txt"
        if repo_train.is_file() and repo_test.is_file():
            official_train, official_test = repo_train, repo_test
    if not official_train.is_file() or not official_test.is_file():
        raise FileNotFoundError(
            f"Expected official train/test ID lists under {config_dir}; install the dataset protocol files."
        )
    official_train_ids = _read_id_list(official_train)
    official_test_ids = _read_id_list(official_test)
    if len(official_train_ids) != len(set(official_train_ids)):
        raise ValueError(f"Duplicate IDs in {official_train}")
    if len(official_test_ids) != len(set(official_test_ids)):
        raise ValueError(f"Duplicate IDs in {official_test}")
    if set(official_train_ids) & set(official_test_ids):
        raise ValueError("Official train and test ID lists overlap.")
    outside_train = (set(train_ids) | set(val_ids)) - set(official_train_ids)
    if outside_train:
        raise ValueError(
            f"Internal train/validation lists contain {len(outside_train)} IDs outside the official train list."
        )
    return {
        "official_train_list_sha256": sha256_file(official_train),
        "official_train_id_count": len(official_train_ids),
        "official_test_list_sha256": sha256_file(official_test),
        "official_test_id_count": len(official_test_ids),
        "official_test_used_for_selection": False,
    }


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(
                f"Run metadata already exists with different contents: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    os.replace(temporary, path)


def write_training_run_metadata(args: Any, *, method: str) -> Path:
    official_protocol = _validate_training_splits(args)
    output_dir = Path(args.save_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir = output_dir / "protocol"
    train_split = _copy_split(
        getattr(args, "train_split_file", None), protocol_dir / "train_ids.txt"
    )
    val_split = _copy_split(
        getattr(args, "val_split_file", None), protocol_dir / "val_ids.txt"
    )
    config_paths = [
        getattr(args, "config_path", None),
        getattr(args, "ng_config_path", None),
    ]
    sources = _source_files(config_paths)
    source_hash_path = output_dir / "source_sha256.txt"
    source_lines = [
        f"{sha256_file(path)}  {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}"
        for path in sources
    ]
    source_content = "\n".join(source_lines) + "\n"
    if source_hash_path.exists() and source_hash_path.read_text(encoding="utf-8") != source_content:
        raise FileExistsError(f"Refusing to overwrite source hashes: {source_hash_path}")
    if not source_hash_path.exists():
        with source_hash_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(source_content)

    try:
        import torch

        torch_info = {
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        torch_info = {"pytorch": None, "cuda_runtime": None, "cuda_available": False, "gpu": None}

    pretrained = None
    model_config = getattr(args, "config_path", None)
    if model_config and Path(model_config).is_file():
        config_values = yaml.safe_load(Path(model_config).read_text(encoding="utf-8")) or {}
        pretrained_value = config_values.get("sam_ckpt_path")
        if pretrained_value:
            pretrained_path = Path(pretrained_value).expanduser()
            if not pretrained_path.is_absolute():
                pretrained_path = REPO_ROOT / pretrained_path
            pretrained = {
                "path": str(pretrained_path),
                "sha256": sha256_file(pretrained_path) if pretrained_path.is_file() else None,
            }

    config = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "dataset": args.dataset,
        "training": {
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epoch,
            "initial_learning_rate": args.lr,
            "seed": args.seed,
            "precision": "fp32",
            "loss": getattr(args, "loss_func", "bceloss"),
            "num_workers": args.num_workers,
            "optimizer": "Adam",
            "optimizer_groups": {
                "detector_lower": {
                    "name": "Adam",
                    "initial_learning_rate": args.lr,
                    "weight_decay": 1e-4,
                },
                "detector_upper": {
                    "name": "Adam",
                    "initial_learning_rate": args.lr * 0.5,
                    "weight_decay": 1e-4,
                },
                "noise_generator": (
                    {
                        "name": "SGD",
                        "initial_learning_rate": args.lr * 0.01,
                        "weight_decay": 1e-6,
                    }
                    if method == "learned_ng_bilevel"
                    else None
                ),
            },
            "minimum_learning_rate": getattr(args, "eta_min", None),
            "checkpoint_interval_epochs": getattr(args, "save_step", None),
        },
        "learned_ng": {
            "enabled": method == "learned_ng_bilevel",
            "strength_scale": getattr(args, "ng_strength_scale", None),
            "start_epoch": getattr(args, "ng_start_epoch", None),
            "end_epoch": getattr(args, "ng_end_epoch", None),
            "bilevel_start_epoch": getattr(args, "bilevel_start_epoch", None),
            "bilevel_end_epoch": getattr(args, "bilevel_end_epoch", None),
            "ng_optimizer": "SGD" if method == "learned_ng_bilevel" else None,
        },
        "protocol": {
            **official_protocol,
            "train_split": train_split,
            "validation_split": val_split,
        },
        "source": {**_git_provenance(), "hash_file": "source_sha256.txt"},
        "pretrained_backbone": pretrained,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            **torch_info,
        },
        "command_line": sys.argv,
        "resume_from": getattr(args, "resume", None),
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists() and getattr(args, "resume", None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        config_path = output_dir / f"resume_config_{stamp}.json"
    elif config_path.exists():
        raise FileExistsError(
            f"Run directory already has metadata: {output_dir}. Choose a new SAVE_DIR, "
            "or pass --resume to continue the recorded run."
        )
    elif not getattr(args, "resume", None) and any(
        list(output_dir.glob("*.pth")) + list(output_dir.glob("*.pt"))
    ):
        raise FileExistsError(
            f"A checkpoint already exists in {output_dir} without a run manifest; "
            "choose a new SAVE_DIR or recover its original run metadata."
        )
    _write_json_once(config_path, config)
    return config_path


def write_evaluation_metadata(args: Any, *, output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights).expanduser().resolve()
    test_list = Path(args.test_split_file).expanduser().resolve() if args.test_split_file else None
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
    if test_list is None or not test_list.is_file():
        raise FileNotFoundError("An explicit matching official test ID list is required.")
    archived_test_list = (
        _copy_split(str(test_list), output_dir / "protocol" / "official_test_ids.txt")
        if test_list and test_list.is_file()
        else None
    )
    source_lines = [
        f"{sha256_file(path)}  {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}"
        for path in _source_files([])
    ]
    source_name = f"evaluation_source_sha256_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.txt"
    with (output_dir / source_name).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(source_lines) + "\n")
    value = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "checkpoint": {"path": str(weights), "sha256": sha256_file(weights)},
        "official_test_list": archived_test_list,
        "test_list_warning": None if test_list and test_list.is_file() else "No explicit test ID list was recorded.",
        "source_hash_file": source_name,
        "command_line": sys.argv,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    config_path = output_dir / f"evaluation_config_{stamp}.json"
    _write_json_once(config_path, value)
    return config_path
