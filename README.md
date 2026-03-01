# Two-Way Is Better Than One (ICLR 2026) — Official Code

This repository contains the official implementation for the ICLR 2026 paper:

**Two-Way Is Better Than One: Bidirectional Alignment with Cycle Consistency for Exemplar-Free Class-Incremental Learning**  
Hongye Xu, Bartosz Krawczyk

OpenReview: https://openreview.net/forum?id=7UfZAxKo5K

---

## Overview

In exemplar-free class-incremental learning (EFCIL), we cannot store past data, so representation drift makes cached class statistics (e.g., prototypes / Gaussians) stale and causes severe forgetting.  
We propose **bidirectional alignment with cycle consistency** during training, jointly learning two lightweight maps:

- **A: old → new** (adapter; transports stored old-class statistics into the current feature space),
- **D: new → old** (distiller; regularizes the current representation toward the previous backbone),

together with **stop-gradient gating** and a **cycle-consistency loss** so that transport and representation co-evolve.

---

## Codebase

Our implementation is **based on the FACIL benchmark**:
- FACIL: https://github.com/mmasana/FACIL

---

## Setup

### 1) Create conda env + install dependencies

```bash
conda create -n yourenv python=3.10 -y
conda activate TT

# PyTorch (CUDA 12.6 wheels)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126

# Core libs
pip install timm==1.0.15 einops==0.8.1 \
  numpy==2.2.5 scipy==1.15.2 pandas==2.2.3 scikit-learn==1.6.1 \
  matplotlib==3.10.0 pillow==11.2.1 tqdm==4.67.1 pyyaml==6.0.2

```

> Notes:
> - For GPU training, ensure your NVIDIA driver supports CUDA 12.6.
> - If you want a minimal dependency set, keep only the “Core libs” and remove the “Optional” block unless required by your run.

---

## Datasets

### 1) Download datasets
Please download datasets following your preferred convention (cluster/shared storage, etc.).

### 2) Set dataset root path
Set the dataset root by editing:

- `src/datasets/dataset_config.py`
  - modify `_BASE_DATA_PATH` to your local dataset root directory.

---

## Reproducing Experiments

We provide scripts under `scripts/`.

### TinyImageNet (10 tasks × 20 classes)
```bash
bash scripts/tiny-10x20.sh
```

### CIFAR-100 (10 tasks × 10 classes)
```bash
bash scripts/cifar-10x10.sh
```

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{xu2026two_way,
  title     = {Two-Way Is Better Than One: Bidirectional Alignment with Cycle Consistency for Exemplar-Free Class-Incremental Learning},
  author    = {Xu, Hongye and Krawczyk, Bartosz},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=7UfZAxKo5K}
}
```

---

## Acknowledgements

This repository is built upon the AdaGauss, FACIL benchmark and related continual-learning tooling. We thank the respective authors for releasing their code.
