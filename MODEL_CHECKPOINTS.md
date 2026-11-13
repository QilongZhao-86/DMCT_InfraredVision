# Ours checkpoint archive

The [checkpoint Release](https://github.com/QilongZhao-86/DMCT_InfraredVision/releases/tag/ours-checkpoints-2026-11-12) contains 39 Ours-labelled checkpoint files in eight dataset and experiment packages. The Release includes per-file SHA-256 values and available run metadata. Dataset images and masks are obtained separately from their providers.

## Small-target detection

The final paper reports 300 training epochs for NUDT-SIRST, IRSTD-1K, and IRSTDID-SKY. The archive preserves completed runs, candidate grids, SKY validation snapshots, and weights originally stored with paper-related filenames. Each file's manifest entry identifies its run, dataset, size, SHA-256, and available evaluation record.

| Dataset / model | Archived run and evaluation | Checkpoint |
|---|---|---|
| NUDT-SIRST Ours, job 6754 | 350 epochs; batch 12; seed 2727; best internal-validation mIoU 94.7192. Recorded evaluation: mIoU 95.13, F1 97.50, nIoU 95.03, Pd 99.05, Fa 0.00029. | `NUDT-SIRST_ours_job6754_legacy350.pth`; SHA-256 `30e1ec5841b961c0377792ad9c4932210557b8b8fbd419fe56506f1a911a452b` |
| IRSTD-1K Ours, job 6791 | 350 epochs; batch 8; seed 2727; best internal-validation mIoU 70.9068 at epoch 257. | `IRSTD-1K_ours_job6791_legacy350.pth`; SHA-256 `3ac1cdb91f6d745b884d92b2a13747dcc98ee4e5fe335de7407a8e62376cc5f4` |
| IRSTDID-SKY baseline, job 6854 | 300 epochs; batch 8; seed 2727; canonical upstream 342/60/98 train/validation/test split. Recorded official-test evaluation: mIoU 68.66. | `IRSTDID-SKY_baseline_job6854_canonical300.pth`; SHA-256 `4a6d5f202ec7dcdc982bbb0e6b1de4581e05dd28bef3d0f296972016afa5683b` |

The NUDT evaluation record identifies the loaded checkpoint, reports zero missing or unexpected keys, and includes 664,000 precision-recall samples. The 300-epoch candidate grids and the 350-epoch runs are separately organized in the Release packages.

The SKY Ours package includes the canonical candidate grid and the job 6857 internal-validation history. Job 6857 used the upstream 402/98 train/test lists and a fixed seed-2727 split of 342 training and 60 validation images. Its highest recorded internal-validation mIoU is 72.3026 at epoch 197. The SKY baseline evaluation and Ours internal-validation results are listed as distinct experiments.

The archive also contains checkpoints originally named for the paper's Ours method. Their recorded evaluations differ from some Table IV values; the manifest preserves the original filenames and the measured run metrics for each file.

## FMB semantic segmentation

The segmentation packages contain the archived 18,000-iteration Ours checkpoint, the robust fine-tuning checkpoint selected at iteration 5,000, and the 20,000-iteration checkpoint under its archived filename. The recorded clean/mild/moderate/severe robust mIoU values are 52.81/52.73/51.90/52.12. The 20,000-iteration checkpoint is indexed by its archived filename in the manifest.

The paper's FMB results are listed in the README. The Release manifest gives the available checkpoint files, run labels, and recorded metrics.
