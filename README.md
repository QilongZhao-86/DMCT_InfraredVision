# Bridging the Thermal Gap: Data-Model Collaborative Transfer for Infrared Perception

**Authors:** Zhu Liu, Qilong Zhao, Zhirui Fang, Benzhuang Zhang, Jinyuan Liu, and Risheng Liu.

This repository accompanies the accepted manuscript *Bridging the Thermal Gap: Data-Model Collaborative Transfer for Infrared Perception*. The paper proposes a data-model collaborative transfer framework for infrared visual perception.

## Method overview

The framework addresses two related challenges: infrared training data cover only a limited range of imaging conditions, and visible-pretrained foundation models do not directly account for thermal image formation. It connects data construction and model adaptation through four components:

1. **Task-responsive data generation** learns weights over plausible infrared variations, including low contrast, blur, sensor and optical noise, and resolution changes.
2. **Formation-guided representation** extracts complementary structure-dominant and response-detail cues from infrared observations.
3. **Thermal-aware model transfer** uses those cues to condition lightweight adapters while keeping the foundation-model backbone frozen.
4. **Single-loop collaborative optimization** updates data generation and model-transfer components together without repeated inner-loop convergence.

The manuscript evaluates the approach on infrared image enhancement, depth estimation, small-target detection, and semantic segmentation.

## Contents of this release

- Infrared model, training, and evaluation code in `infrared_model/`, `training/`, `dataset/`, `train_final.py`, and `training.py`.
- Baseline and learned noise-generator/bilevel training entry points.
- Portable Linux launch examples in `scripts/`, with paths supplied through environment variables.
- Fixed train/validation IDs and official IRSTDID-SKY test IDs in `reproducibility/`.
- Archived checkpoint inventory and SHA-256 records in `MODEL_CHECKPOINTS.md` and `CHECKPOINT_SHA256SUMS.txt`.
- Dataset acquisition notes, environment notes, and third-party notices.

This snapshot contains the infrared small-target detection pipeline and its archived results. It does not include the complete implementations and artifacts for all four task instantiations described in the manuscript. Dataset images and masks are not included.

## Archived small-target detection checkpoints

These are the archived runs available in this code release. Their training histories and evaluation protocols are recorded in [`MODEL_CHECKPOINTS.md`](MODEL_CHECKPOINTS.md). They are artifact-specific results and should not be substituted for the manuscript's results.

| Dataset / model | Archived training record | Evaluation status |
|---|---|---|
| NUDT-SIRST model | Legacy 350-epoch run; batch 12; seed 2727 | One recorded evaluation: mIoU 95.13, F1 97.50, nIoU 95.03, Pd 99.05, Fa 2.90 |
| IRSTD-1K model | Legacy 350-epoch run; batch 8; seed 2727; best internal-validation mIoU 70.9068 | No official-test evaluation recorded |
| IRSTDID-SKY baseline | 300 epochs; batch 8; seed 2727; canonical 342/60/98 train/validation/test split | One recorded official evaluation: mIoU 68.66 |

The NUDT and IRSTD-1K checkpoints are legacy 350-epoch runs, not results from the later 300-epoch candidate grids. Incomplete runs and noncanonical SKY results are excluded. Do not compare results across different splits, training budgets, or checkpoint-selection protocols.

## Setup

Use Linux with a CUDA-capable PyTorch installation. `environment.yml` records the environment used for the archived runs; it is not a cross-platform lockfile. See [`ENVIRONMENT.md`](ENVIRONMENT.md).

Download the SAM 2.0 Hiera Tiny pretrained checkpoint from the official SAM 2 release and place it at:

```text
infrared_model/sam2_ckpt/sam2_hiera_tiny.pt
```

Pretrained SAM 2 weights and datasets are not included. Do not commit them to the source repository. The selected trained checkpoints are local release assets under `release-assets/checkpoints/` and are excluded from ordinary Git commits; if redistribution is approved, attach them to a GitHub Release and verify them against `CHECKPOINT_SHA256SUMS.txt`.

## Dataset layout

Set `DATA_ROOT` to a directory containing one subdirectory per dataset, with `images/` and `masks/` under each dataset directory. The fixed train/validation ID files are provided in `reproducibility/<dataset>/`. Obtain datasets from their official providers and follow their terms. See [`DATASETS.md`](DATASETS.md).

## Training

The scripts use environment variables for paths and contain no machine-specific server paths. Example baseline run:

```bash
DATA_ROOT=/datasets/IRSTD \
SPLIT_DIR=reproducibility/NUDT-SIRST \
DATASET=NUDT-SIRST \
BATCH_SIZE=12 IMAGE_SIZE=256 EPOCHS=300 LR=0.01 SEED=2727 \
SAVE_DIR=outputs/nudt-baseline \
bash scripts/train_baseline.sh
```

Example learned noise-generator with bilevel training:

```bash
DATA_ROOT=/datasets/IRSTD \
SPLIT_DIR=reproducibility/IRSTDID-SKY \
DATASET=IRSTDID-SKY \
BATCH_SIZE=8 IMAGE_SIZE=512 EPOCHS=300 LR=0.005 SEED=2727 \
NG_STRENGTH_SCALE=0.25 SAVE_DIR=outputs/sky-ours \
bash scripts/train_ours.sh
```

The learned noise-generator example starts NG and bilevel optimization at epoch 25 and retains the trainer's existing loss, alpha schedule, and update equations. The examples do not submit cluster jobs; adapt them to your scheduler if needed.

## Evaluation

Select a checkpoint using the fixed internal validation split, then lock it before running evaluation on its matching official test list. Do not use official-test results for checkpoint or hyperparameter selection.

```bash
DATA_ROOT=/datasets/IRSTD \
DATASET=IRSTDID-SKY IMAGE_SIZE=512 BATCH_SIZE=8 SEED=2727 \
WEIGHTS=release-assets/checkpoints/IRSTDID-SKY_baseline_job6854_canonical300.pth \
TEST_LIST=reproducibility/IRSTDID-SKY/official_test_ids.txt \
RESULTS_DIR=outputs/sky-baseline-test \
bash scripts/evaluate_checkpoint.sh
```

## Citation

Zhu Liu, Qilong Zhao, Zhirui Fang, Benzhuang Zhang, Jinyuan Liu, and Risheng Liu, “Bridging the Thermal Gap: Data-Model Collaborative Transfer for Infrared Perception.” Accepted manuscript. The supplied PDF does not include final publication venue, volume, pages, or DOI; add those details from the publisher's version when available.

## License and third-party code

Third-party notices are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). No project license has been assigned here. Before making the repository public or redistributing code/checkpoints, confirm authorization from the relevant copyright holders and complete [`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md).
