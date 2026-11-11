# SAM-SPL: Infrared Small-Target Detection

Research code for training and evaluating SAM-SPL infrared small-target
detectors. This repository contains the model implementation, data loader,
training/evaluation entry points, portable shell examples, and fixed split ID
files. Dataset images and masks are not included.

## Contents

- `training.py`: baseline training.
- `train_final.py`: learned-noise-generator and bilevel training.
- `test_pr_audit.py`: checkpoint evaluation with an explicit test-list option.
- `sam_spl/`, `training/`, and `dataset/`: model, optimization, and data code.
- `reproducibility/`: fixed train/validation IDs and the canonical SKY lists.
- `SPLIT_SHA256SUMS.txt`: split counts and list-file hashes.
- `scripts/`: environment-variable-driven Linux launch examples.
- `MODEL_CHECKPOINTS.md`: completed archived checkpoint inventory and metrics.
- `CHECKPOINT_SHA256SUMS.txt`: hashes for the model release assets.

## Requirements

Use Linux with a CUDA-capable PyTorch installation. The supplied environment
file records the environment used for archived runs; it is not a cross-platform
lockfile. See [`ENVIRONMENT.md`](ENVIRONMENT.md) for setup notes.

Download the SAM 2.0 Hiera Tiny pretrained checkpoint from the official SAM 2
release and place it at:

```text
sam_spl/sam2_ckpt/sam2_hiera_tiny.pt
```

Do not commit pretrained model weights to the source repository. The archived
SAM-SPL checkpoints are provided as separate release assets under
`release-assets/checkpoints/` for local review; upload them to a GitHub Release
as separate files if their publication is approved.

## Dataset layout

Set `DATA_ROOT` to a directory containing one subdirectory per dataset. Each
dataset directory must contain `images/` and `masks/`. Training and validation
ID files are supplied in `reproducibility/<dataset>/`; the dataset itself must
be obtained from its official provider. See [`DATASETS.md`](DATASETS.md).

## Training

The scripts require paths through environment variables and contain no
machine-specific server paths. Example for the baseline:

```bash
DATA_ROOT=/datasets/IRSTD \
SPLIT_DIR=reproducibility/NUDT-SIRST \
DATASET=NUDT-SIRST \
BATCH_SIZE=12 IMAGE_SIZE=256 EPOCHS=300 LR=0.01 SEED=2727 \
SAVE_DIR=outputs/nudt-baseline \
bash scripts/train_baseline.sh
```

Example for learned NG with bilevel training:

```bash
DATA_ROOT=/datasets/IRSTD \
SPLIT_DIR=reproducibility/IRSTDID-SKY \
DATASET=IRSTDID-SKY \
BATCH_SIZE=8 IMAGE_SIZE=512 EPOCHS=300 LR=0.005 SEED=2727 \
NG_STRENGTH_SCALE=0.25 SAVE_DIR=outputs/sky-ours \
bash scripts/train_ours.sh
```

The learned-NG example starts NG and bilevel optimization at epoch 25 and
retains the trainer's existing loss, alpha schedule, and update equations.
The examples do not submit cluster jobs; adapt them to your scheduler if
needed.

## Evaluation

Evaluate only after locking a checkpoint. Supply the official test ID file
explicitly through `TEST_LIST`; do not use official-test results for model or
hyperparameter selection.

```bash
DATA_ROOT=/datasets/IRSTD \
DATASET=IRSTDID-SKY IMAGE_SIZE=512 BATCH_SIZE=8 SEED=2727 \
WEIGHTS=release-assets/checkpoints/IRSTDID-SKY_baseline_job6854_canonical300.pth \
TEST_LIST=reproducibility/IRSTDID-SKY/official_test_ids.txt \
RESULTS_DIR=outputs/sky-baseline-test \
bash scripts/evaluate_checkpoint.sh
```

## Checkpoints and reported results

The release-asset directory contains three completed small-target checkpoints.
Each result is labeled with its dataset protocol and checkpoint hash in
[`MODEL_CHECKPOINTS.md`](MODEL_CHECKPOINTS.md). Historical 300-epoch sweeps,
partial runs, canceled jobs, and noncanonical SKY results are not part of the
release inventory.

## Citation and license

This code release is identified as the SAM-SPL implementation. The repository
does not claim to be the complete artifact for a separately titled manuscript.
Add the final paper citation after confirming the exact accepted title and
author-approved citation.

Third-party source notices are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
The project license must be approved by the code copyright holders before
public upload; no license is assigned here on their behalf.
