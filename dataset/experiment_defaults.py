"""Paper-aligned defaults for the supported infrared detection datasets."""

DATASET_ALIASES = {
    "NUDT-SIRST": "NUDT-SIRST",
    "IRSTD-1K": "IRSTD-1K",
    "IRSTD-1k": "IRSTD-1K",
    "IRSTDID-SKY": "IRSTDID-SKY",
}

TRAINING_DEFAULTS = {
    "NUDT-SIRST": {"image_size": 256, "batch_size": 12},
    "IRSTD-1K": {"image_size": 512, "batch_size": 8},
    "IRSTDID-SKY": {"image_size": 512, "batch_size": 8},
}


def get_training_defaults(dataset: str) -> dict[str, int]:
    """Return resolution and batch defaults, rejecting unknown dataset names."""
    canonical = DATASET_ALIASES.get(dataset)
    if canonical is None:
        supported = ", ".join(TRAINING_DEFAULTS)
        raise ValueError(f"Unsupported dataset {dataset!r}; choose one of: {supported}")
    return TRAINING_DEFAULTS[canonical].copy()
