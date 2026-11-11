# SAM-SPL: Infrared Small-Target Detection

**Paper authors:** Zhu Liu, Qilong Zhao, Zhirui Fang, Benzhuang Zhang, Jinyuan Liu, and Risheng Liu.

This repository releases **SAM-SPL, our model for infrared small-target detection**, together with its training and evaluation code, fixed data splits, and selected checkpoints. It accompanies the accepted manuscript *Bridging the Thermal Gap: Data-Model Collaborative Transfer for Infrared Perception*.

> **Release scope:** SAM-SPL is our model, not a third-party baseline in this code release. This repository provides its infrared small-target detection implementation and reproducibility materials. It does not currently contain complete implementations and artifacts for every task discussed in the paper (infrared enhancement, depth estimation, small-target detection, and semantic segmentation). The checkpoint results below describe the archived runs included with this repository; consult the paper for its reported results and protocols.

## Paper overview

Infrared perception is affected by a modality gap between visible-light pretraining and thermal observations, as well as changing sensor and environmental conditions. The paper proposes to address these issues jointly at the data and model levels:

1. **Task-responsive data generation** learns how to combine differentiable, physically plausible infrared variations, including low contrast, blur, sensor noise, optical noise, and resolution changes.
2. **Formation-guided representation** derives complementary structure-dominant and response-detail cues from infrared observations.
3. **Thermal-aware model transfer** uses those cues to condition lightweight adapters while keeping the foundation-model backbone frozen.
4. **Single-loop collaborative optimization** updates data generation and model-transfer components together without requiring repeated inner-loop convergence.

The manuscript evaluates the framework on infrared image enhancement, depth estimation, small-target detection, and semantic segmentation. See the paper for its complete method, protocols, comparisons, and results.

## SAM-SPL model and released code

- Our SAM-SPL model and training code in `sam_spl/`, `training/`, `dataset/`, `train_final.py`, and `training.py`.
- Baseline and learned noise-generator/bilevel training entry points.
- Environment-variable-based Linux examples in `scripts/`; no lab-server paths are embedded in these scripts.
- Fixed train/validation IDs and the canonical IRSTDID-SKY official test IDs in `reproducibility/`.
- Archived checkpoint inventory and SHA-256 records in `MODEL_CHECKPOINTS.md` and `CHECKPOINT_SHA256SUMS.txt`.
- Dataset acquisition notes, environment notes, and third-party notices.

Datasets are not included. The current release does not include the full enhancement, depth-estimation, or FMB semantic-segmentation pipelines described in the manuscript.

## Archived detection checkpoints

These are artifact-specific records, with their own training histories and evaluation status. See [`MODEL_CHECKPOINTS.md`](MODEL_CHECKPOINTS.md) for the full protocol and hashes.

| Dataset / model | Archived training record | Evaluation status |
|---|---|---|
| NUDT-SIRST Ours | Legacy 350-epoch run; batch 12; seed 2727 | One recorded evaluation: mIoU 95.13, F1 97.50, nIoU 95.03, Pd 99.05, Fa 2.90 |
| IRSTD-1K Ours | Legacy 350-epoch run; batch 8; seed 2727; best internal-validation mIoU 70.9068 | No official-test evaluation recorded |
| IRSTDID-SKY baseline | 300 epochs; batch 8; seed 2727; canonical 342/60/98 train/validation/test split | One recorded official evaluation: mIoU 68.66 |

The NUDT and IRSTD-1K checkpoints are legacy 350-epoch runs, not results from the later 300-epoch candidate grids. Incomplete runs and noncanonical SKY results are excluded. Do not compare or combine results across different splits, training budgets, or checkpoint-selection protocols.

## Setup

Use Linux with a CUDA-capable PyTorch installation. `environment.yml` records the environment used for the archived runs; it is not a cross-platform lockfile. See [`ENVIRONMENT.md`](ENVIRONMENT.md).

Download the SAM 2.0 Hiera Tiny pretrained checkpoint from the official SAM 2 release and place it at:

```text
sam_spl/sam2_ckpt/sam2_hiera_tiny.pt
```

Pretrained SAM 2 weights and datasets are not included in this repository. Do not commit them here. The selected SAM-SPL checkpoints are local release assets under `release-assets/checkpoints/` and are excluded from ordinary Git commits; if redistribution is approved, attach them as files to a GitHub Release and verify them against `CHECKPOINT_SHA256SUMS.txt`.

## Dataset layout

Set `DATA_ROOT` to a directory containing one subdirectory per dataset, with `images/` and `masks/` under each dataset directory. The fixed train/validation ID files are provided in `reproducibility/<dataset>/`. Obtain each dataset from its official provider and follow its terms. See [`DATASETS.md`](DATASETS.md).

## Training

The scripts use environment variables for paths and do not contain machine-specific server paths. Example baseline run:

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

The learned-NG example starts NG and bilevel optimization at epoch 25 and retains the trainer's existing loss, alpha schedule, and update equations. The examples do not submit cluster jobs; adapt them to your scheduler.

## Evaluation

Select a checkpoint using the fixed internal validation split, then lock it before running the matching official test list. Do not use official-test results for checkpoint or hyperparameter selection.

```bash
DATA_ROOT=/datasets/IRSTD \
DATASET=IRSTDID-SKY IMAGE_SIZE=512 BATCH_SIZE=8 SEED=2727 \
WEIGHTS=release-assets/checkpoints/IRSTDID-SKY_baseline_job6854_canonical300.pth \
TEST_LIST=reproducibility/IRSTDID-SKY/official_test_ids.txt \
RESULTS_DIR=outputs/sky-baseline-test \
bash scripts/evaluate_checkpoint.sh
```

## Paper citation

Zhu Liu, Qilong Zhao, Zhirui Fang, Benzhuang Zhang, Jinyuan Liu, and Risheng Liu, “Bridging the Thermal Gap: Data-Model Collaborative Transfer for Infrared Perception.” Accepted manuscript. The supplied PDF does not include final publication venue, volume, pages, or DOI; add those details from the publisher's version when available.

## License and third-party code

Third-party notices are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). A project license has not been assigned in this repository. Before making the repository public or redistributing code/checkpoints, confirm authorization from the relevant copyright holders and complete [`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md).
