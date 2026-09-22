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
- [Data Preparation](#data-preparation)
- [Leaderboard](#leaderboard)
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

| File | Size | Download |
|:-----|:----:|:--------:|
| `occlusion_label_train.json` | 802 KB | [Download](https://github.com/wenbo-wei/COCO-OLAC/releases/download/annotations-v1.0/occlusion_label_train.json) |
| `occlusion_label_val.json` | 114 KB | [Download](https://github.com/wenbo-wei/COCO-OLAC/releases/download/annotations-v1.0/occlusion_label_val.json) |
| `occlusion_label_val_low.json` | 30 KB | [Download](https://github.com/wenbo-wei/COCO-OLAC/releases/download/annotations-v1.0/occlusion_label_val_low.json) |
| `occlusion_label_val_mid.json` | 55 KB | [Download](https://github.com/wenbo-wei/COCO-OLAC/releases/download/annotations-v1.0/occlusion_label_val_mid.json) |
| `occlusion_label_val_high.json` | 49 KB | [Download](https://github.com/wenbo-wei/COCO-OLAC/releases/download/annotations-v1.0/occlusion_label_val_high.json) |

## Annotation Format

Each annotation file is a single JSON object that maps a COCO `image_id` (zero-padded to 12 digits, encoded as a string) to one of the three occlusion levels:

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

## Data Preparation

Follow Mask2Former's dataset preparation [instructions](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md) to generate the required semantic and panoptic annotations in advance. Set `DETECTRON2_DATASETS=datasets/data` so that Detectron2 resolves dataset paths correctly. Place the dataset under `datasets/data/`, following the structure below:

```
datasets/data/
  coco_olac/
    annotations/
      instances_{train,val,val_low,val_mid,val_high}.json
      panoptic_{train,val,val_low,val_mid,val_high}.json
    {train,val,val_low,val_mid,val_high}/  # RGB images
    panoptic_{train,val,val_low,val_mid,val_high}/
    panoptic_semseg_{train,val,val_low,val_mid,val_high}/
    occlusion_label_{train,val,val_low,val_mid,val_high}.json
```

## Leaderboard

Each method is evaluated using its **official pre-trained weights** on the per-level validation subsets (paper Table I).

| Method            | Occlusion        | PQ                 | PQ<sup>Th</sup>    | PQ<sup>St</sup>    | AP<sub>pan</sub><sup>Th</sup> | mIoU<sub>pan</sub> |
|:------------------|:----------------:|:------------------:|:------------------:|:------------------:|:-----------------------------:|:------------------:|
| Panoptic&nbsp;FPN      | l/m/h            | 43.8&nbsp;/&nbsp;40.2&nbsp;/&nbsp;34.5 | 53.2&nbsp;/&nbsp;47.3&nbsp;/&nbsp;39.0 | 29.5&nbsp;/&nbsp;29.5&nbsp;/&nbsp;27.7 | —                             | —                  |
| Panoptic&nbsp;FCN      | l/m/h            | 46.9&nbsp;/&nbsp;44.9&nbsp;/&nbsp;36.3 | 56.1&nbsp;/&nbsp;48.2&nbsp;/&nbsp;40.4 | 33.3&nbsp;/&nbsp;32.5&nbsp;/&nbsp;30.1 | —                             | —                  |
| Panoptic&nbsp;DeepLab  | l/m/h            | 42.9&nbsp;/&nbsp;36.2&nbsp;/&nbsp;30.0 | 47.8&nbsp;/&nbsp;39.4&nbsp;/&nbsp;31.0 | 35.5&nbsp;/&nbsp;31.3&nbsp;/&nbsp;29.2 | —                             | —                  |
| MaskFormer        | l/m/h            | 52.6&nbsp;/&nbsp;48.0&nbsp;/&nbsp;41.2 | 58.3&nbsp;/&nbsp;53.9&nbsp;/&nbsp;44.0 | 43.3&nbsp;/&nbsp;39.1&nbsp;/&nbsp;37.0 | —                             | —                  |
| Mask2Former       | l/m/h            | 56.8&nbsp;/&nbsp;53.3&nbsp;/&nbsp;46.7 | 64.4&nbsp;/&nbsp;60.1&nbsp;/&nbsp;51.3 | 45.8&nbsp;/&nbsp;43.0&nbsp;/&nbsp;39.7 | 56.5&nbsp;/&nbsp;45.1&nbsp;/&nbsp;35.8            | 60.4&nbsp;/&nbsp;61.2&nbsp;/&nbsp;58.1 |
| Mask&nbsp;DINO         | l/m/h            | 56.6&nbsp;/&nbsp;53.7&nbsp;/&nbsp;48.3 | 63.1&nbsp;/&nbsp;60.6&nbsp;/&nbsp;53.3 | 47.0&nbsp;/&nbsp;43.4&nbsp;/&nbsp;40.8 | 56.4&nbsp;/&nbsp;47.2&nbsp;/&nbsp;38.8            | 58.0&nbsp;/&nbsp;59.7&nbsp;/&nbsp;57.4 |

**Note:** `l`, `m`, and `h` denote low, mid, and high occlusion levels, respectively.

## Citation

If you find the COCO-OLAC dataset useful in your research, please cite our paper:

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

The code in this repository is released under <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" align="absmiddle"></a>. The newly introduced occlusion-level annotations are released under **CC-BY 4.0**, consistent with the underlying COCO images. <!-- TODO: confirm CC-BY-4.0 is the intended license for the new labels -->
