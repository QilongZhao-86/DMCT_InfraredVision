# Completed checkpoint inventory

The files in `release-assets/checkpoints/` are local release assets and are
excluded from ordinary Git commits. Each file is identified by SHA-256. Run
evaluation only with the matching dataset and protocol.

| Dataset / model | Training and selection | Evaluation recorded | Checkpoint |
|---|---|---|---|
| NUDT-SIRST Ours, job 6754 | 350 epochs; batch 12; seed 2727; best internal-validation mIoU 94.7192 | One recorded official evaluation: mIoU 95.13, F1 97.50, nIoU 95.03, Pd 99.05, Fa 2.90 | `NUDT-SIRST_ours_job6754_legacy350.pth`; SHA-256 `30e1ec5841b961c0377792ad9c4932210557b8b8fbd419fe56506f1a911a452b` |
| IRSTD-1K Ours, job 6791 | 350 epochs; batch 8; seed 2727; best internal-validation mIoU 70.9068 at epoch 257 | No official-test evaluation recorded | `IRSTD-1K_ours_job6791_legacy350.pth`; SHA-256 `3ac1cdb91f6d745b884d92b2a13747dcc98ee4e5fe335de7407a8e62376cc5f4` |
| IRSTDID-SKY baseline, job 6854 | 300 epochs; batch 8; seed 2727; canonical upstream 342/60/98 train/validation/test protocol | One recorded official evaluation: mIoU 68.66 | `IRSTDID-SKY_baseline_job6854_canonical300.pth`; SHA-256 `4a6d5f202ec7dcdc982bbb0e6b1de4581e05dd28bef3d0f296972016afa5683b` |

The NUDT official-evaluation status is resolved from the archived `eval.log`:
the checkpoint loaded with zero missing/unexpected keys, evaluated 664,000 PR
samples, and printed the metrics above. The earlier note saying that evaluation
was still pending was incorrect.

The NUDT and IRSTD-1K weights are completed legacy 350-epoch runs. They are not
results from the later 300-epoch candidate grids. The SKY row uses the
corrected official lists. No partial/interim SKY Ours checkpoint, old-split
SKY checkpoint, or segmentation-only weight is included in this release.
