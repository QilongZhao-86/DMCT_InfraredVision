<div align="center">

# Bridging the Thermal Gap: Data-Model Collaborative Transfer for Infrared Perception

**Zhu Liu · Qilong Zhao · Zhirui Fang · Benzhuang Zhang · Jinyuan Liu · Risheng Liu**

[![Paper](https://img.shields.io/badge/Paper-Accepted-2563EB?style=for-the-badge)](#citation)
[![Task](https://img.shields.io/badge/Domain-Infrared%20Vision-0891B2?style=for-the-badge)](#method-overview)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=for-the-badge)](#training)
[![Reproducibility](https://img.shields.io/badge/Artifacts-Reproducibility-7C3AED?style=for-the-badge)](#checkpoints)

**A data-model collaborative framework for robust infrared visual perception.**

[Method](#method-overview) · [Results](#paper-reported-results) · [Installation](#setup) · [Training](#training) · [Citation](#citation)

<img src="assets/framework_overview.png" alt="Overview of the data-model collaborative transfer framework" width="100%">

*Overview of the framework: task-responsive data generation, formation-guided infrared representation, and thermal-aware model transfer, evaluated across four infrared perception tasks.*

</div>

## 📢 News

- The manuscript has been accepted. Publication venue and DOI will be added when final publisher metadata are available.
- This repository currently releases the infrared small-target detection pipeline and its selected reproducibility artifacts.

## ✨ Method overview

Infrared perception must handle imaging variations such as low contrast, blur, sensor and optical noise, and changes in resolution. The paper couples training-data construction with model transfer through four components:

- **Task-responsive data generation** learns weights over plausible infrared imaging variations to construct informative observations.
- **Formation-guided representation** derives complementary structure-dominant and response-detail cues from thermal images.
- **Thermal-aware model transfer** uses those cues to condition lightweight adapters while keeping the foundation-model backbone frozen.
- **Single-loop collaborative optimization** updates data generation and model transfer together without repeated inner-loop convergence.

The paper evaluates the framework on **infrared image enhancement, depth estimation, small-target detection, and semantic segmentation**.

## 📊 Paper-reported results

The following values are transcribed from the paper's Table IV (the proposed method, “Ours”). They are manuscript results, not measurements of a checkpoint currently distributed with this repository.

| Dataset | Method | mIoU ↑ | F1 ↑ | Pd ↑ | Fa ↓ (×10⁻⁴) |
|:--|:--:|--:|--:|--:|--:|
| NUDT-SIRST | **Ours** | **95.26** | **97.57** | **99.26** | **0.0064** |
| IRSTD-1K | **Ours** | **66.40** | **79.81** | **92.26** | **0.2019** |
| IRSTDID-SKY | **Ours** | **72.65** | **84.16** | **98.72** | **0.0079** |
| Overall | **Ours** | **78.10** | **87.18** | **96.75** | **0.0743** |

> **Checkpoint status:** The final checkpoints corresponding to these paper results have not yet been supplied for this release. They will be added with their matching protocol details and SHA-256 hashes when available. The separate [`MODEL_CHECKPOINTS.md`](MODEL_CHECKPOINTS.md) file documents earlier archived runs; those weights and metrics are not the source of the table above.

This release contains the small-target detection code and artifacts. It does not yet include complete implementations and artifacts for all four task instantiations described in the manuscript. Dataset images and masks are not included.

## 🧰 Contents

- Model and training code: `infrared_model/`, `training/`, `dataset/`, `train_final.py`, and `training.py`.
- Portable Linux launch examples: `scripts/`.
- Fixed train/validation IDs and official IRSTDID-SKY test IDs: `reproducibility/`.
- Checkpoint inventory and hashes: `MODEL_CHECKPOINTS.md`, `CHECKPOINT_SHA256SUMS.txt`.
- Dataset and environment guidance: [`DATASETS.md`](DATASETS.md), [`ENVIRONMENT.md`](ENVIRONMENT.md).
- Third-party notices: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## ⚙️ Setup

Use Linux with a CUDA-capable PyTorch installation. `environment.yml` records the environment used for archived runs; it is not a cross-platform lockfile. See [`ENVIRONMENT.md`](ENVIRONMENT.md) for setup details.

Download the SAM 2.0 Hiera Tiny pretrained checkpoint from the official SAM 2 release and place it at:

```text
infrared_model/sam2_ckpt/sam2_hiera_tiny.pt
```

Pretrained SAM 2 weights and datasets are not included. Do not commit them to the source repository. The selected trained checkpoints are local release assets under `release-assets/checkpoints/` and are excluded from ordinary Git commits. If redistribution is approved, attach them to a GitHub Release and verify them against `CHECKPOINT_SHA256SUMS.txt`.

## 🗂️ Dataset layout

Set `DATA_ROOT` to a directory containing one subdirectory per dataset, with `images/` and `masks/` under each dataset directory. Fixed train/validation ID files are supplied in `reproducibility/<dataset>/`. Obtain datasets from their official providers and follow their terms. See [`DATASETS.md`](DATASETS.md).

## 🚀 Training

The launch scripts take paths from environment variables and contain no machine-specific server paths.

**Baseline**

```bash
DATA_ROOT=/datasets/IRSTD \
SPLIT_DIR=reproducibility/NUDT-SIRST \
DATASET=NUDT-SIRST \
BATCH_SIZE=12 IMAGE_SIZE=256 EPOCHS=300 LR=0.01 SEED=2727 \
SAVE_DIR=outputs/nudt-baseline \
bash scripts/train_baseline.sh
```

**Learned noise generator + bilevel training**

```bash
DATA_ROOT=/datasets/IRSTD \
SPLIT_DIR=reproducibility/IRSTDID-SKY \
DATASET=IRSTDID-SKY \
BATCH_SIZE=8 IMAGE_SIZE=512 EPOCHS=300 LR=0.005 SEED=2727 \
NG_STRENGTH_SCALE=0.25 SAVE_DIR=outputs/sky-ours \
bash scripts/train_ours.sh
```

The learned-NG example starts NG and bilevel optimization at epoch 25 and retains the trainer's existing loss, alpha schedule, and update equations. The scripts do not submit cluster jobs; adapt them to your scheduler.

## 🔬 Evaluation

Select a checkpoint using the fixed internal-validation split, then lock it before evaluating on its matching official test list. Do not use official-test results for checkpoint or hyperparameter selection.

```bash
DATA_ROOT=/datasets/IRSTD \
DATASET=IRSTDID-SKY IMAGE_SIZE=512 BATCH_SIZE=8 SEED=2727 \
WEIGHTS=release-assets/checkpoints/IRSTDID-SKY_baseline_job6854_canonical300.pth \
TEST_LIST=reproducibility/IRSTDID-SKY/official_test_ids.txt \
RESULTS_DIR=outputs/sky-baseline-test \
bash scripts/evaluate_checkpoint.sh
```

## 📚 Citation

Zhu Liu, Qilong Zhao, Zhirui Fang, Benzhuang Zhang, Jinyuan Liu, and Risheng Liu, “Bridging the Thermal Gap: Data-Model Collaborative Transfer for Infrared Perception.” Accepted manuscript. The supplied PDF does not include the final publication venue, volume, pages, or DOI; please add those details from the publisher's version when available.

## 📄 License and third-party code

Third-party notices are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). No project license has been assigned here. Before making the repository public or redistributing code/checkpoints, confirm authorization from the relevant copyright holders and complete [`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md).
