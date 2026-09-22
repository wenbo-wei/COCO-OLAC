<div align="center">

# COCO-OLAC: A Benchmark for Occluded Panoptic Segmentation and Image Understanding

[Wenbo Wei](https://github.com/wenbo-wei), [Jun Wang](https://github.com/Markin-Wang), [Abhir Bhalerao](https://scholar.google.com/citations?hl=en&user=XfBoSP4AAAAJ)

[![Conference](https://img.shields.io/badge/ICASSP-2025-blue.svg)](https://arxiv.org/abs/2409.12760)
[![Dataset](https://img.shields.io/badge/Dataset-COCO--OLAC-orange.svg)](#download)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Annotations](https://img.shields.io/badge/Annotations-CC--BY--4.0-green.svg)](#license)

</div>

## Table of Contents

- [News](#news)
- [Introduction](#introduction)
- [Highlights](#highlights)
- [Statistics](#statistics)
- [Download](#download)
- [Annotation Format](#annotation-format)
- [Benchmark Protocol](#benchmark-protocol)
- [Leaderboard](#leaderboard)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## News

- [2025-04] Paper published at **ICASSP 2025**.

## Introduction

<p align="justify">
To help address the occlusion problem in panoptic segmentation and image understanding, this paper proposes a new large-scale dataset named COCO-OLAC (COCO Occlusion Labels for All Computer Vision Tasks), which is derived from the COCO dataset by manually labelling images into three perceived occlusion levels. Using COCO-OLAC, we systematically assess and quantify the impact of occlusion on panoptic segmentation on samples having different levels of occlusion. Comparative experiments with SOTA panoptic models demonstrate that the presence of occlusion significantly affects performance, with higher occlusion levels resulting in notably poorer performance. Additionally, we propose a straightforward yet effective method as an initial attempt to leverage the occlusion annotation using contrastive learning to render a model that learns a more robust representation capturing different severities of occlusion. Experimental results demonstrate that the proposed approach boosts the performance of the baseline model and achieves SOTA performance on the proposed COCO-OLAC dataset.
</p>

## Highlights

- **Occlusion-level annotations.** 35,000 COCO images are manually labelled into three perceived occlusion levels: low, mid, and high.
- **Fine-grained evaluation.** Per-level validation subsets support robustness analysis across different occlusion severities.
- **Unified benchmark.** Six representative panoptic segmentation models are evaluated under a common protocol, revealing consistent performance degradation as occlusion increases.
- **Occlusion-aware baseline.** A contrastive learning baseline uses the occlusion labels to learn more robust feature representations.

## Statistics

| Split | Source                                          | Total  | Low   | Mid    | High   |
|:------|:------------------------------------------------|:------:|:-----:|:------:|:------:|
| Train | First 30,000 images of COCO `train2017`         | 30,000 | 6,668 | 11,251 | 12,081 |
| Val   | Full COCO `val2017`                             | 5,000  | 1,134 | 2,075  | 1,791  |

## Download

**Occlusion-level annotations** are distributed directly in this repository at the project root:

| File                              | Size   | Contents                                                       |
|:----------------------------------|:------:|:---------------------------------------------------------------|
| `occlusion_label_train.json`      | 802 KB | 30,000 training images, three-level labels                     |
| `occlusion_label_val.json`        | 114 KB | 5,000 validation images, three-level labels                    |
| `occlusion_label_val_low.json`    | 30 KB  | low occlusion subset (1,134 images)                            |
| `occlusion_label_val_mid.json`    | 55 KB  | mid occlusion subset (2,075 images)                            |
| `occlusion_label_val_high.json`   | 49 KB  | high occlusion subset (1,791 images)                           |

**Images and panoptic masks** are *not* redistributed. Please obtain the official **COCO 2017** images and panoptic annotations from <https://cocodataset.org/#download> and place them under `datasets/data/coco_olac/` as described in [Data preparation](#data-preparation).

## Annotation Format

Each file `occlusion_label_{train,val}.json` is a single JSON object that maps a COCO `image_id` (zero-padded to 12 digits, encoded as a string) to one of the three occlusion levels:

```json
{
  "000000432898": "high",
  "000000461009": "high",
  "000000246436": "high",
  "000000397133": "mid",
  "000000037777": "low",
  "...": "..."
}
```

Levels follow the manual annotation protocol defined in the paper (Sec. II.A):

| Level  | Occluded-region ratio | Definition                                            |
|:-------|:---------------------:|:------------------------------------------------------|
| `low`  | 0%                    | No perceivable occlusion in the scene                 |
| `mid`  | 0–50%                 | Partial occlusion of one or more foreground objects   |
| `high` | 50–100%               | Severe occlusion of at least one foreground object    |

A reference script for regenerating and splitting the labels is provided at `tools/create_eval_occl_label.py`.

## Benchmark Protocol

### Data preparation

Place the data under `datasets/data/`:

```
datasets/data/
└── coco_olac/
    ├── train/                                    # first 30k images of COCO train2017
    ├── val/, val_low/, val_mid/, val_high/        # full validation set and per-level subsets
    ├── panoptic_{train,val,val_low,val_mid,val_high}/
    ├── panoptic_semseg_{train,val,val_low,val_mid,val_high}/
    ├── annotations/
    │   ├── instances_{train,val,val_low,val_mid,val_high}.json
    │   └── panoptic_{train,val,val_low,val_mid,val_high}.json
    └── occlusion_label_{train,val,val_low,val_mid,val_high}.json
```

The split image and mask directories share one dataset root; all instance and
panoptic JSONs share `annotations/`. Braces denote separate names.

### Evaluation splits

Models are reported on the following partitions of the validation set:

- **Full val** (5,000 images) — overall metric.
- **Val-Low** (1,134), **Val-Mid** (2,075), and **Val-High** (1,791) — per-level subsets.

### Metrics

Standard panoptic metrics are reported throughout:

- **PQ** — Panoptic Quality, reported as overall, *thing*, and *stuff* variants.
- **AP<sub>pan</sub><sup>Th</sup>** — instance AP derived from panoptic predictions on *thing* classes.
- **mIoU<sub>pan</sub>** — semantic mean IoU derived from panoptic predictions.

The corresponding evaluation utilities are provided at `tools/evaluate_pq_for_semantic_segmentation.py` and `tools/evaluate_coco_boundary_ap.py`.

## Leaderboard

### Validation experiment (paper Table I)

Each method is evaluated using its **official pre-trained weights** on the per-level validation subsets; no fine-tuning on COCO-OLAC is performed.

| Method            | Occlusion        | PQ                 | PQ<sup>Th</sup>    | PQ<sup>St</sup>    | AP<sub>pan</sub><sup>Th</sup> | mIoU<sub>pan</sub> |
|:------------------|:----------------:|:------------------:|:------------------:|:------------------:|:-----------------------------:|:------------------:|
| Panoptic FPN      | l/m/h            | 43.8 / 40.2 / 34.5 | 53.2 / 47.3 / 39.0 | 29.5 / 29.5 / 27.7 | —                             | —                  |
| Panoptic FCN      | l/m/h            | 46.9 / 44.9 / 36.3 | 56.1 / 48.2 / 40.4 | 33.3 / 32.5 / 30.1 | —                             | —                  |
| Panoptic DeepLab  | l/m/h            | 42.9 / 36.2 / 30.0 | 47.8 / 39.4 / 31.0 | 35.5 / 31.3 / 29.2 | —                             | —                  |
| MaskFormer        | l/m/h            | 52.6 / 48.0 / 41.2 | 58.3 / 53.9 / 44.0 | 43.3 / 39.1 / 37.0 | —                             | —                  |
| Mask2Former       | l/m/h            | 56.8 / 53.3 / 46.7 | 64.4 / 60.1 / 51.3 | 45.8 / 43.0 / 39.7 | 56.5 / 45.1 / 35.8            | 60.4 / 61.2 / 58.1 |
| Mask DINO         | l/m/h            | 56.6 / 53.7 / 48.3 | 63.1 / 60.6 / 53.3 | 47.0 / 43.4 / 40.8 | 56.4 / 47.2 / 38.8            | 58.0 / 59.7 / 57.4 |

## Installation

The implementation builds on **Mask2Former** (Meta, MIT) and **detectron2**. Please follow the upstream [Mask2Former installation guide](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md) for the heavy dependencies (PyTorch, detectron2, the MSDeformAttn CUDA operator), and then install the remaining requirements:

```bash
pip install -r requirements.txt
```

<!-- TODO: write install_env.sh once env is pinned -->

## Training

```bash
bash scripts/train_conocc_olac_r50.sh
```

This launches `train_net.py` with `configs/coco_olac/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml` and the contrastive hyper-parameters specified in the paper (margins τ<sub>l,h</sub>=0.4, τ<sub>m</sub>=0.6, λ=1.0).

## Evaluation

```bash
bash scripts/eval_conocc_olac_r50.sh
```

By default the script expects the checkpoint at `output/coco_olac/res50/con/model_final.pth`; the path may be overridden via `MODEL.WEIGHTS <path>`.

## Citation

If you find the COCO-OLAC dataset, the proposed evaluation protocol, or the reference implementation useful in your research, please cite our paper:

<!-- TODO: replace with the final published reference once available -->

```bibtex
@inproceedings{wei2025coco,
  title        = {{COCO-OLAC}: A Benchmark for Occluded Panoptic Segmentation and Image Understanding},
  author       = {Wei, Wenbo and Wang, Jun and Bhalerao, Abhir},
  booktitle    = {ICASSP 2025 -- 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages        = {1--5},
  year         = {2025},
  organization = {IEEE}
}
```

## Acknowledgements

COCO-OLAC builds upon the following open-source projects and datasets:

- [COCO](https://cocodataset.org) — source images and original panoptic annotations.
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) (Meta, MIT) — panoptic segmentation framework; modified portions retain the original Meta copyright headers.
- [detectron2](https://github.com/facebookresearch/detectron2) (Meta, Apache 2.0) — training engine and data loading.
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) (SenseTime, Apache 2.0) — MSDeformAttn CUDA operator used by Mask2Former.

We thank the authors of these works for releasing their code and data.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The code in this repository is released under the **MIT license**. The newly introduced occlusion-level annotations are released under **CC-BY 4.0**, consistent with the underlying COCO images. <!-- TODO: confirm CC-BY-4.0 is the intended license for the new labels -->
