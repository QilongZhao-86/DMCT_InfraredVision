# Completed checkpoint inventory

The files in `release-assets/checkpoints/` are local release assets and are
excluded from ordinary Git commits. Each file is identified by SHA-256. Run
evaluation only with the matching dataset and protocol.

Additional Ours-labelled archives are published under the GitHub Release
[`ours-checkpoints-2026-11-12`](https://github.com/QilongZhao-86/DMCT_InfraredVision/releases/tag/ours-checkpoints-2026-11-12).
That release contains historical candidate runs, SKY internal-validation
snapshots, paper-named checkpoint archives, and FMB segmentation weights. Its
per-file manifest and the SHA-256 of each release asset are included with the
release files. The artifacts are provided for preservation and inspection;
they are not all final-paper checkpoints.

These checkpoints are historical artifacts, not the checkpoints behind the
paper's Table IV. The archived NUDT and IRSTD-1K Ours runs used 350 epochs,
while the manuscript specifies 300. Their records do not include a complete
per-run source/config snapshot for every optimizer detail. The current
documented training entrypoints use Adam to match the manuscript; do not claim
that these older weights were produced by the current Adam configuration.

| Dataset / model | Training and selection | Evaluation recorded | Checkpoint |
|---|---|---|---|
| NUDT-SIRST Ours, job 6754 | 350 epochs; batch 12; seed 2727; best internal-validation mIoU 94.7192 | One recorded official evaluation: mIoU 95.13, F1 97.50, nIoU 95.03, Pd 99.05, Fa 2.90 | `NUDT-SIRST_ours_job6754_legacy350.pth`; SHA-256 `30e1ec5841b961c0377792ad9c4932210557b8b8fbd419fe56506f1a911a452b` |
| IRSTD-1K Ours, job 6791 | 350 epochs; batch 8; seed 2727; best internal-validation mIoU 70.9068 at epoch 257 | No official-test evaluation recorded | `IRSTD-1K_ours_job6791_legacy350.pth`; SHA-256 `3ac1cdb91f6d745b884d92b2a13747dcc98ee4e5fe335de7407a8e62376cc5f4` |
| IRSTDID-SKY baseline, job 6854 | 300 epochs; batch 8; seed 2727; canonical upstream 342/60/98 train/validation/test protocol | One recorded official evaluation: mIoU 68.66 | `IRSTDID-SKY_baseline_job6854_canonical300.pth`; SHA-256 `4a6d5f202ec7dcdc982bbb0e6b1de4581e05dd28bef3d0f296972016afa5683b` |

The NUDT official-evaluation status is resolved from the archived `eval.log`:
the checkpoint loaded with zero missing/unexpected keys, evaluated 664,000 PR
samples, and printed the metrics above. The earlier note saying that evaluation
was still pending was incorrect.

The NUDT and IRSTD-1K weights listed above are completed legacy 350-epoch runs.
They are not results from the later 300-epoch candidate grids. The SKY row uses
the corrected official lists and is a baseline, not Ours. SKY Ours candidates
and internal-validation snapshots are available in the separate weights
Release linked above; they are not official-test-selected paper-result weights.

The repository's paper-named checkpoint archives are also included as
historical files in that Release. Their provenance is not fully reconciled with
the accepted manuscript: archived evaluations differ from some Table IV
values. Do not treat their filenames as proof that they generated the paper
results. The release notes and per-weight manifest preserve this caveat.
