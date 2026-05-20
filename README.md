<div align="center">

# COCO-OLAC: A Benchmark for Occluded Panoptic Segmentation and Image Understanding

[Wenbo Wei](https://github.com/wenbo-wei), [Jun Wang](https://github.com/Markin-Wang), [Abhir Bhalerao](https://scholar.google.com/citations?hl=en&user=XfBoSP4AAAAJ)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Annotations](https://img.shields.io/badge/Annotations-CC--BY--4.0-green.svg)](#license)
[![Conference](https://img.shields.io/badge/ICASSP-2025-blue.svg)](#citation)
[![Dataset](https://img.shields.io/badge/Dataset-COCO--OLAC-orange.svg)](#download)

</div>

---

## Overview

COCO-OLAC is released as a benchmark for the diagnostic evaluation of panoptic segmentation under realistic occlusion. The labels, splits, and evaluation protocol form the primary deliverable; the included reference implementation is provided to demonstrate that the new annotations are useful as a training signal.

- **Three perceived occlusion levels** are manually annotated on COCO images, defined by the occluded-region ratio: low (0%), mid (0–50%), and high (50–100%).
- **Per-level validation subsets** support fine-grained robustness analysis of any panoptic, detection, or segmentation model.
- **Six panoptic baselines are benchmarked** under a unified protocol — Panoptic FPN/FCN/DeepLab, MaskFormer, Mask2Former, and Mask DINO — revealing a consistent drop in PQ as occlusion severity rises.
- **An occlusion-aware contrastive learning baseline** is provided as a reference implementation.

## Table of Contents

- [Statistics](#statistics)
- [Download](#download)
- [Annotation Format](#annotation-format)
- [Benchmark Protocol](#benchmark-protocol)
- [Leaderboard](#leaderboard)
- [Reference Implementation](#reference-implementation-contrastive-learning-on-occlusion-levels)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Statistics

| Split | Source                                          | Total  | Low   | Mid    | High   |
|:------|:------------------------------------------------|:------:|:-----:|:------:|:------:|
| Train | First 30,000 images of COCO `train2017`         | 30,000 | 6,668 | 11,251 | 12,081 |
| Val   | Full COCO `val2017`                             | 5,000  | 1,134 | 2,075  | 1,791  |

The validation set is additionally partitioned into three per-level subsets, so that any model may be evaluated independently on low-, mid-, and high-occlusion images.

## Download

**Occlusion-level annotations** are distributed directly in this repository at the project root:

| File                              | Size   | Contents                                                       |
|:----------------------------------|:------:|:---------------------------------------------------------------|
| `occlusion_label_train.json`      | 802 KB | 30,000 training images, three-level labels                     |
| `occlusion_label_val.json`        | 114 KB | 5,000 validation images, three-level labels                    |
| `occlusion_label_val_low.json`    | 30 KB  | Validation subset — low occlusion only (1,134 images)          |
| `occlusion_label_val_mid.json`    | 55 KB  | Validation subset — mid occlusion only (2,075 images)          |
| `occlusion_label_val_high.json`   | 49 KB  | Validation subset — high occlusion only (1,791 images)         |

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
    ├── train2017/                                 # first 30k images of COCO train2017
    ├── val2017/                                   # full COCO val2017
    ├── annotations/
    │   ├── panoptic_train2017.json, panoptic_val2017.json
    │   └── panoptic_{train,val}2017/              # PNG panoptic masks
    └── occlusion_label_{train,val}.json
```

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
| Panoptic FPN      | low / mid / high | 43.8 / 40.2 / 34.5 | 53.2 / 47.3 / 39.0 | 29.5 / 29.5 / 27.7 | —                             | —                  |
| Panoptic FCN      | low / mid / high | 46.9 / 44.9 / 36.3 | 56.1 / 48.2 / 40.4 | 33.3 / 32.5 / 30.1 | —                             | —                  |
| Panoptic DeepLab  | low / mid / high | 42.9 / 36.2 / 30.0 | 47.8 / 39.4 / 31.0 | 35.5 / 31.3 / 29.2 | —                             | —                  |
| MaskFormer        | low / mid / high | 52.6 / 48.0 / 41.2 | 58.3 / 53.9 / 44.0 | 43.3 / 39.1 / 37.0 | —                             | —                  |
| Mask2Former       | low / mid / high | 56.8 / 53.3 / 46.7 | 64.4 / 60.1 / 51.3 | 45.8 / 43.0 / 39.7 | 56.5 / 45.1 / 35.8            | 60.4 / 61.2 / 58.1 |
| Mask DINO         | low / mid / high | 56.6 / 53.7 / 48.3 | 63.1 / 60.6 / 53.3 | 47.0 / 43.4 / 40.8 | 56.4 / 47.2 / 38.8            | 58.0 / 59.7 / 57.4 |

**Observation.** PQ decreases monotonically from low to high occlusion across every method, confirming that the manual annotation captures a meaningful difficulty signal.

### Retraining experiment (paper Table II)

Each method is **retrained on the 30k COCO-OLAC training set** under identical hyper-parameters and subsequently evaluated on the full validation set.

| Model                                          | PQ       | PQ<sup>Th</sup> | PQ<sup>St</sup> | AP<sub>pan</sub><sup>Th</sup> | mIoU<sub>pan</sub> |
|:-----------------------------------------------|:--------:|:---------------:|:---------------:|:-----------------------------:|:------------------:|
| Panoptic FPN                                   | 33.3     | 39.1            | 24.6            | —                             | —                  |
| Panoptic DeepLab                               | 27.5     | 28.9            | 25.3            | —                             | —                  |
| Panoptic FCN                                   | 33.0     | 37.7            | 26.0            | —                             | —                  |
| Mask2Former                                    | 41.5     | 45.3            | 35.6            | 30.6                          | **54.4**           |
| YOSO                                           | 37.1     | 40.6            | 31.9            | —                             | —                  |
| **Ours** (Mask2Former + contrastive baseline)  | **41.8** | **45.3**        | **36.4**        | **30.8**                      | 54.3               |

Submissions to the leaderboard are welcome — please open a pull request with the per-level numbers and a link to the trained checkpoint.

## Reference Implementation: Contrastive Learning on Occlusion Levels

In addition to the benchmark, we release a simple baseline that **uses the occlusion labels at training time**. A triplet contrastive loss is applied in the backbone feature space, pulling together representations of images at the same occlusion level and pushing apart those at different levels.

<p align="center"><img src="assets/architecture.png" width="700" alt="Contrastive baseline pipeline"></p>

Per-level improvements over the retrained Mask2Former baseline (paper Table III):

| Occlusion | Model    | PQ              | PQ<sup>Th</sup> | PQ<sup>St</sup> | AP<sub>pan</sub><sup>Th</sup> | mIoU<sub>pan</sub> |
|:---------:|:---------|:---------------:|:---------------:|:---------------:|:-----------------------------:|:------------------:|
| Low       | Baseline | 47.5            | 53.5            | 38.5            | 46.7                          | 52.2               |
|           | Ours     | **48.1** (+0.6) | 53.1            | **40.8** (+2.3) | 46.4                          | **54.0** (+0.5)    |
| Mid       | Baseline | 43.1            | 48.1            | 35.6            | 33.3                          | 54.0               |
|           | Ours     | **43.2**        | 47.9            | **36.1** (+0.5) | **33.7** (+0.4)               | 54.0               |
| High      | Baseline | 35.7            | 38.2            | 32.0            | 24.8                          | 50.7               |
|           | Ours     | **36.1** (+0.4) | 38.2            | **33.0** (+1.0) | 24.7                          | **50.8**           |

### Installation

The implementation builds on **Mask2Former** (Meta, MIT) and **detectron2**. Please follow the upstream [Mask2Former installation guide](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md) for the heavy dependencies (PyTorch, detectron2, the MSDeformAttn CUDA operator), and then install the remaining requirements:

```bash
pip install -r requirements.txt
```

<!-- TODO: write install_env.sh once env is pinned -->

### Training

```bash
bash scripts/train_conocc_olac_r50.sh
```

This launches `train_net.py` with `configs/coco_olac/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml` and the contrastive hyper-parameters specified in the paper (margins τ<sub>l,h</sub>=0.4, τ<sub>m</sub>=0.6, λ=1.0).

### Evaluation

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
