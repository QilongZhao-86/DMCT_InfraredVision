<div align="center">

# Bridging the Thermal Gap: Data-Model Collaborative Transfer for Infrared Perception

**Zhu Liu · Qilong Zhao · Zhirui Fang · Benzhuang Zhang · Jinyuan Liu · Risheng Liu**

[![Paper](https://img.shields.io/badge/TPAMI-Regular%20Paper-Accepted-2563EB?style=for-the-badge)](#citation)
[![Task](https://img.shields.io/badge/Domain-Infrared%20Vision-0891B2?style=for-the-badge)](#method-overview)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?style=for-the-badge)](#training)
[![Reproducibility](https://img.shields.io/badge/Artifacts-Reproducibility-7C3AED?style=for-the-badge)](#checkpoints)

**A data-model collaborative framework for robust infrared visual perception.**

[Method](#method-overview) · [Results](#paper-reported-results) · [Installation](#setup) · [Training](#training) · [Citation](#citation)

<img src="assets/framework_overview.png" alt="Overview of the data-model collaborative transfer framework" width="100%">

*Overview of the framework: task-responsive data generation, formation-guided infrared representation, and thermal-aware model transfer, evaluated across four infrared perception tasks.*

</div>

## 📢 News

- **6 Nov 2026:** Accepted as a Regular Paper in *IEEE Transactions on Pattern Analysis and Machine Intelligence* (manuscript **TPAMI-2026-07-2953.R1**). DOI and final volume/issue/page or article-number details will be added after publication.
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

### Infrared semantic segmentation on FMB

The final manuscript also reports semantic segmentation on FMB using DINOv2, with AdamW at an initial learning rate of `1e-4`. The adapter is inserted after transformer blocks, with frequency decomposition applied in feature space. Table V evaluates ×2 and ×4 resolution degradation; Table VI evaluates moderate and severe stripe noise. The manuscript does not specify batch size, random seed, or training epochs for this task. The values below are the paper's reported mIoU percentages, not results reproduced from assets in this release.

| FMB evaluation setting | Method | mIoU (%) |
|:--|:--:|--:|
| ×2 scaling factor | **Ours** | **52.03** |
| ×4 scaling factor | **Ours** | **51.13** |
| Moderate stripe noise | **Ours** | **50.93** |
| Severe stripe noise | **Ours** | **50.56** |

This repository release does not include the complete FMB segmentation implementation, its trained checkpoint, or the data. The table records manuscript results only.

## 🧪 Small-target detection configuration

The manuscript's small-target detection setup specifies the following shared training settings. False-alarm rate values in the paper are reported in units of `10⁻⁴`.

| Dataset | Input resolution | Training | Optimizer | Initial learning rate |
|:--|:--:|:--:|:--:|--:|
| NUDT-SIRST | 256 × 256 | 300 epochs | Adam | 0.01 |
| IRSTD-1K | 512 × 512 | 300 epochs | Adam | 0.01 |
| IRSTDID-SKY | 512 × 512 | 300 epochs | Adam | 0.01 |

The manuscript does not report batch size or random seed. The commands below use the repository's recorded launch settings (batch size 12 for NUDT-SIRST, 8 for IRSTD-1K and IRSTDID-SKY; seed 2727 for the included fixed split manifests). These are release settings, not values attributed to the paper.

The detector's lower and upper bilevel parameter groups use Adam in this release; the upper group retains the trainer's existing half-rate setting. The noise generator keeps its separate SGD optimizer. This change aligns the detector optimizer family with the manuscript while preserving the existing bilevel losses, alpha schedule, and update equations. The earlier checkpoints predate this optimizer alignment, and their per-run source/config records are not included here. Do not present them as Adam-based reproductions or as the weights that produced the manuscript's Table IV results.

Other implementation-specific hyperparameters (including noise-generator scaling and save cadence) are not specified in the manuscript. The launch scripts accept explicit overrides for these values; their defaults are implementation choices, not paper-reported settings.

This release contains the small-target detection code and artifacts. It does not yet include complete implementations and artifacts for all four task instantiations described in the manuscript. Dataset images and masks are not included.

## 🧰 Contents

- Model and training code: `infrared_model/`, `training/`, `dataset/`, `train_final.py`, and `training.py`.
- Paper-aligned dataset defaults: `dataset/experiment_defaults.py`.
- Per-run config, split, and source-hash capture: `training/run_metadata.py`.
- Portable Linux launch examples: `scripts/`.
- Fixed train/validation IDs and official IRSTDID-SKY test IDs: `reproducibility/`.
- Checkpoint inventory and hashes: `MODEL_CHECKPOINTS.md`, `CHECKPOINT_SHA256SUMS.txt`.
- Dataset and environment guidance: [`DATASETS.md`](DATASETS.md), [`ENVIRONMENT.md`](ENVIRONMENT.md).
- Citation metadata: `CITATION.cff`.
- Third-party notices: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## ⚙️ Setup

Use Linux with a CUDA-capable PyTorch installation. `environment.yml` records the environment used for archived runs; it is not a cross-platform lockfile. See [`ENVIRONMENT.md`](ENVIRONMENT.md) for setup details.

Download the SAM 2.0 Hiera Tiny pretrained checkpoint from the official SAM 2 release and place it at:

```text
infrared_model/sam2_ckpt/sam2_hiera_tiny.pt
```

Pretrained SAM 2 weights and datasets are not included. Do not commit them to the source repository. The selected trained checkpoints are local release assets under `release-assets/checkpoints/` and are excluded from ordinary Git commits. If redistribution is approved, attach them to a GitHub Release and verify them against `CHECKPOINT_SHA256SUMS.txt`.

## 🗂️ Dataset layout

Set `DATA_ROOT` to the dataset package root. It must contain each canonical dataset directory with `images/` and `masks/`, plus the provider's `set_configs/<dataset>/train.txt` and `test.txt` lists for NUDT-SIRST and IRSTD-1K. SKY's canonical lists are supplied in `reproducibility/IRSTDID-SKY/`. Fixed train/validation ID files are supplied in `reproducibility/<dataset>/`. Obtain image and mask data from official providers and follow their terms. See [`DATASETS.md`](DATASETS.md).

## 🚀 Training

The launch scripts take paths from environment variables and contain no machine-specific server paths. They choose resolution and batch defaults from the dataset name, and accept explicit overrides. The examples use the paper-reported resolution, epoch count, detector optimizer, and initial learning rate.

**NUDT-SIRST**

```bash
DATA_ROOT=/datasets/IRSTD \
SPLIT_DIR=reproducibility/NUDT-SIRST \
DATASET=NUDT-SIRST \
BATCH_SIZE=12 IMAGE_SIZE=256 EPOCHS=300 LR=0.01 SEED=2727 \
SAVE_DIR=outputs/nudt-ours \
bash scripts/train_ours.sh
```

**IRSTD-1K**

```bash
DATA_ROOT=/datasets/IRSTD \
SPLIT_DIR=reproducibility/IRSTD-1K \
DATASET=IRSTD-1K \
BATCH_SIZE=8 IMAGE_SIZE=512 EPOCHS=300 LR=0.01 SEED=2727 \
SAVE_DIR=outputs/irstd1k-ours \
bash scripts/train_ours.sh
```

**IRSTDID-SKY**

```bash
DATA_ROOT=/datasets/IRSTD \
SPLIT_DIR=reproducibility/IRSTDID-SKY \
DATASET=IRSTDID-SKY \
BATCH_SIZE=8 IMAGE_SIZE=512 EPOCHS=300 LR=0.01 SEED=2727 \
SAVE_DIR=outputs/sky-ours \
bash scripts/train_ours.sh
```

The learned-NG training script starts NG and bilevel optimization at epoch 25 and retains the trainer's existing loss, alpha schedule, and update equations. The scripts do not submit cluster jobs; adapt them to your scheduler. For a paper-config run, start from the pretrained SAM 2 backbone and a fresh optimizer state; the archived checkpoints' optimizer provenance is not recorded. The supported reproductions are the `train_final.py`/`scripts/train_ours.sh` path and the plain-detector `training.py`/`scripts/train_baseline.sh` comparison path. Other trainer files are retained as experimental implementations and are not the documented reproduction entrypoints.

Each training run writes `run_config.json`, `source_sha256.txt`, and exact copies of its train/validation ID lists under `SAVE_DIR`. Reusing a populated run directory for a new run is rejected; use a new `SAVE_DIR` (or `--resume` for a recorded Ours run). The official test list is read only to check split disjointness; its images and masks are not loaded during training or checkpoint selection.

## 🔬 Evaluation

Select a checkpoint using the fixed internal-validation split, then lock it before evaluating on its matching official test list. Do not use official-test results for checkpoint or hyperparameter selection.

```bash
DATA_ROOT=/datasets/IRSTD \
DATASET=IRSTDID-SKY IMAGE_SIZE=512 BATCH_SIZE=8 SEED=2727 \
WEIGHTS=/path/to/sky_ours_checkpoint.pth \
TEST_LIST=reproducibility/IRSTDID-SKY/official_test_ids.txt \
RESULTS_DIR=outputs/sky-ours-test \
bash scripts/evaluate_checkpoint.sh
```

The evaluator writes a timestamped `evaluation_config_*.json`, a source-hash manifest, and a copy of the exact test ID list under `RESULTS_DIR`. It also records the evaluated checkpoint's SHA-256.

## 📚 Citation

Zhu Liu, Qilong Zhao, Zhirui Fang, Benzhuang Zhang, Jinyuan Liu, and Risheng Liu, “Bridging the Thermal Gap: Data-Model Collaborative Transfer for Infrared Perception,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, accepted as a Regular Paper, manuscript TPAMI-2026-07-2953.R1. DOI and final bibliographic details are pending publication.

## 📄 License and third-party code

Third-party notices are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). No project license has been assigned here. Before making the repository public or redistributing code/checkpoints, confirm authorization from the relevant copyright holders and complete [`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md).
