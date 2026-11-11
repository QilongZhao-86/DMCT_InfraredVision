# Environment

`environment.yml` records the Linux/CUDA packages from the archived research
environment. Its host-specific Conda prefix has been removed. It is a starting
point for setup, not a guarantee that a different GPU driver, CUDA runtime, or
Linux distribution will reproduce the archived environment exactly.

Install PyTorch and torchvision builds compatible with the machine's CUDA
driver, then install the remaining packages from the Conda file. Download the
SAM 2.0 Hiera Tiny checkpoint with `bash scripts/download_sam2_tiny.sh`.

The publication preparation has not completed a clean-machine training run.
Record the OS, Python, PyTorch, CUDA, GPU, command line, split hashes, source
hash, and output checkpoint hash for any new reproduction.
