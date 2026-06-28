# Dataset Distillation as Data Compression (DD-RUO) [ICCV 2025]

**English | [简体中文](README.md)**

This repository is the official PyTorch implementation of the paper **Dataset Distillation as Data Compression: A Rate-Utility Perspective** (ICCV 2025).

| [Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Bao_Dataset_Distillation_as_Data_Compression_A_Rate-Utility_Perspective_ICCV_2025_paper.pdf) | [Project page](https://github.com/nouise/DD-RUO) | [Weights](#pretrained-weights) |

## Overview
<!-- ![Teaser image](overview.png) -->
> We recast dataset distillation as a **rate-utility optimization** problem: a synthetic dataset is no longer stored as pixels, but parameterized in a neural image-compression fashion — each (group of) synthetic sample(s) is represented by a set of latent grids, an entropy network and a decoder (following the C3 design). Storage cost is measured in bits-per-pixel (bpp) and utility by downstream training accuracy; their trade-off is controlled by a coefficient λ. This parameterization is orthogonal to the distillation objective and can be uniformly combined with three losses: Trajectory Matching (TM), Gradient Matching (GM), and Distribution Matching (DM).

## Repository structure
```
core/           shared library (one copy for all three methods): the TensorPool representation backbone (latent + entropy network + decoder) + utils and network definitions
TM/             Trajectory Matching: distillation entry pool_tm.py + expert-trajectory generation buffer.py + scripts/
DM/             Distribution Matching: distillation entry pool_dm.py + scripts/
DC/             Gradient Matching (called GM in the paper): distillation entry pool_dc.py + scripts/
quantize/       Stage 2 · post-quantization (shared by all three methods): quantize_pool.py + scripts/
cross_eval/     Stage 3 · cross-architecture evaluation (shared): cross_evaluate.py + scripts/
entropy_codec/  the actual bitstream encode/decode & rate analysis: encode_v2.py / decode_v2.py / analyze.py + scripts/
```
> The three distillation losses share one TensorPool pipeline; they differ only in the distillation entry point (`pool_*.py`). `quantize/` and `cross_eval/` are independent of the distillation method — a pool produced by any method can be used directly.

## Requirements
```
conda env create -f TM/scripts/environment.yml
conda activate dd_ruo
```

## Dataset
Download [ImageNet-1K](https://www.image-net.org/) and fill in the dataset path at the top of each script. Experiments are run on the 128×128 ImageNet 10-class subsets: ImageNette / ImageWoof / ImageFruit / ImageYellow / ImageMeow / ImageSquawk.

## Method
Training a synthetic dataset has four stages (consistent with the paper):

1. **Initialization** (warm-up): initialize the compression representation with the rate-distortion coefficient **β** (TM uses β=10, GM/DM use β=10⁶).
2. **Joint rate-utility optimization**: jointly optimize the distillation objective and the bit-rate with the rate-utility coefficient **λ**, using a **two-stage schedule** — a larger λ in the first half favors distillation performance, then a smaller λ in the second half tightens the bit budget.
3. **Post-quantization**: among a set of MSE thresholds, pick the quantization precision that meets the target bpc budget while being rate-utility optimal.
4. **Evaluation**: train several classifiers and report the average accuracy; cross-architecture evaluation serves as an ablation.

Main hyperparameters:

- `β` — the rate-distortion trade-off at the initialization stage
- `λ` — the rate-utility trade-off at the joint optimization stage (two-stage)
- decoder version / entropy-network context size / slice size — the compression network structure

Detailed values for each dataset and each distillation loss are in the paper; the exact per-experiment launch scripts, log locations and weight mapping are released together with the weights, under [`checkpoints_release/`](https://www.modelscope.cn/models/yiping03/dd-ruo0) in the ModelScope repo.

## Usage
The three distillation losses share one image-representation pipeline; they differ only in the distillation training entry point. When `pool_path=init` in a script, Initialization (warm-up) is performed automatically.

### TM (default)
TM first needs expert trajectories:
```
cd TM/scripts
bash run_buffer.sh
bash run_pool_tm.sh
```

### DM
```
cd DM/scripts
bash run_pool_dm.sh
```

### GM (code directory is `DC/`)
```
cd DC/scripts
bash run_pool_dc.sh
```

### Post-quantization (shared by all three methods)
```
cd quantize/scripts
bash run_quantize.sh
```

### Cross-architecture Evaluation (ablation, shared by all three methods)
```
cd cross_eval/scripts
bash run_cross_eval.sh
```

> The hyperparameters of each stage (β, λ, decoder structure, MSE thresholds, evaluation architectures, etc.) are at the top of the corresponding bash script; please fill them in following the script comments.

## Pretrained weights
The pretrained synthetic datasets, training logs and launch scripts are released on [ModelScope: yiping03/dd-ruo0](https://www.modelscope.cn/models/yiping03/dd-ruo0). After downloading, put them into `checkpoints/` to directly run Post-quantization and Evaluation:

```python
from modelscope import snapshot_download
snapshot_download('yiping03/dd-ruo0')
```

The exact per-experiment launch scripts, log locations and weight mapping are under [`checkpoints_release/`](https://www.modelscope.cn/models/yiping03/dd-ruo0) inside the ModelScope repo.

## Citation
If this work is helpful to your research, please cite:

```bibtex
@inproceedings{bao2025ruo,
  author    = {Youneng Bao and Yiping Liu and Zhuo Chen and Yongsheng Liang and Mu Li and Kede Ma},
  title     = {Dataset Distillation as Data Compression: A Rate-Utility Perspective},
  booktitle = {IEEE/CVF International Conference on Computer Vision},
  year      = {2025},
}
```

## Acknowledgement
This implementation is built on top of the following works:
- *C3: High-performance and low-complexity neural compression (Cool-Chic)* — [Code](https://github.com/Orange-OpenSource/Cool-Chic)
- *Frequency Domain-based Dataset Distillation (FreD)*, NeurIPS 2023 — [Code](https://github.com/sdh0818/FreD)
- *Distilling Dataset into Neural Field (DDiF)*, ICLR 2025 — [Code](https://github.com/aailab-kaist/DDiF)
- *Dataset Condensation with Gradient Matching / Distribution Matching* — [Code](https://github.com/VICO-UoE/DatasetCondensation)
- *Dataset Distillation by Matching Training Trajectories* — [Code](https://github.com/georgecazenavette/mtt-distillation)
