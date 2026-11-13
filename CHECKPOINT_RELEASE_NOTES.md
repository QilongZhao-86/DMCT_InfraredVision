# Ours checkpoint archive

The GitHub Release provides 39 Ours checkpoint files across eight packages, with a per-file SHA-256 manifest and available run metadata. Dataset images and masks are obtained separately from their providers.

## Small-target detection

The packages cover NUDT-SIRST, IRSTD-1K, and IRSTDID-SKY. They include the candidate runs, completed archived runs, SKY internal-validation snapshots, and weights retained under their original paper-related filenames. The per-file manifest lists each checkpoint's dataset, run, size, SHA-256, and recorded evaluation.

The final paper configuration uses 300 training epochs for these datasets. The archive also retains earlier runs at 350 epochs, short smoke runs, and partial runs; their run records identify the training setup used for each checkpoint.

The SKY job 6857 package uses the upstream 402/98 train/test lists and the fixed seed-2727 split of 342 training and 60 validation images. Its highest recorded internal-validation mIoU is 72.3026 at epoch 197. The SKY baseline result and Ours internal-validation results are catalogued as distinct experiments.

The package includes weights originally named for the paper's Ours method. The manifest lists their recorded evaluations alongside the original filenames; some recorded metrics differ from Table IV.

## FMB semantic segmentation

The segmentation packages include the archived 18,000-iteration Ours checkpoint, the robust fine-tuning checkpoint selected at iteration 5,000, and the 20,000-iteration checkpoint under its archived filename. The robust run's recorded clean/mild/moderate/severe mIoU values are 52.81/52.73/51.90/52.12. Per-file run labels and checksums appear in the manifest.

## Reproducibility

Each checkpoint file has a SHA-256 entry. Run configurations, fixed split IDs, source hashes, and available validation or test metrics are grouped with the corresponding archive package.
