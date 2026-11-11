# Reproducibility files

`train_ids.txt` and `val_ids.txt` are the fixed seed-2727 training and internal
validation IDs used by the archived protocol for each dataset. They do not
contain image or mask data.

For IRSTDID-SKY, `official_train_ids.txt` and `official_test_ids.txt` reproduce
the upstream SAM-SPL list files. The internal validation list is derived only
from the official training IDs; the official test list is kept separate.

For NUDT-SIRST and IRSTD-1K, the archived train/validation IDs are included.
The original dataset packages provide their official test lists. Supply the
matching official list through `TEST_LIST` only after the model is locked.
Each split manifest is generated locally by `make_manifest.py` and records the
SHA-256 of the included list files.
