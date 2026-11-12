# Datasets and split protocol

Dataset images and masks are not included. Obtain NUDT-SIRST, IRSTD-1K, and
IRSTDID-SKY from their official providers and follow each provider's terms.
`DATA_ROOT` must contain the canonical dataset directories with `images/` and
`masks/` subdirectories. For NUDT-SIRST and IRSTD-1K, it must also contain the
provider's official ID lists at `set_configs/<canonical-name>/train.txt` and
`test.txt`. SKY's upstream lists are included in
`reproducibility/IRSTDID-SKY/`. Training and validation ID lists are supplied
under `reproducibility/`.

The included SKY protocol uses the upstream 402-ID training list and 98-ID
official test list. A seed-2727 holdout reserves 60 IDs from the official
training list for internal validation, leaving 342 training IDs. The official
test IDs must only be used for one final evaluation after checkpoint selection.

NUDT-SIRST and IRSTD-1K include the fixed archived train/validation IDs. Their
official test ID lists are supplied with the respective dataset packages.
Always evaluate a checkpoint against its matching dataset list and training
protocol; do not tune with test-set metrics.
