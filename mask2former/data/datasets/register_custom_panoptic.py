# Copyright (c) Facebook, Inc. and its affiliates.
import os
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager


_PREDEFINED_SPLITS_COCO_OLAC = {
    "custom": (
        "coco_olac_custom/custom",
        "coco_olac_custom/panoptic_custom",
        "coco_olac_custom/panoptic_semseg_custom",
        "coco_olac_custom/annotations/instances.json",
        "coco_olac_custom/annotations/panoptic.json",
    ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories] to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def load_coco_olac_annos(image_root, panoptic_root, semantic_root, panoptic_json, metadata):
    """
    Args:
        image_root (str): path to the raw dataset.
        panoptic_root (str): path to the raw annotations.
        semantic_root (str): path to the panoptic semantic segmentation annotations.
        panoptic_json (str): path to the panoptic json file.
        metadata (dict): a dictionary maps contiguous id from [0, #thing categories + #stuff categories] to their names and colors.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format.
    """
    
    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(panoptic_json) as pj:
        panoptic_info = json.load(pj)

    ret = []
    for ann in panoptic_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_root, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(panoptic_root, ann["file_name"])
        sem_label_file = os.path.join(semantic_root, ann["file_name"])
        segments_info = [_convert_category_id(x, metadata) for x in ann["segments_info"]]

        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )

    assert len(ret), f"No images found in {image_root}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret


def register_coco_olac_panoptic(
    dataset_name: str,
    metadata: dict,
    image_root: str,
    panoptic_root: str,
    semantic_root: str,
    instances_json: str,
    panoptic_json: str,
):
    
    DatasetCatalog.register(
        dataset_name,
        lambda: load_coco_olac_annos(image_root, panoptic_root, semantic_root, panoptic_json, metadata),
    )
    
    MetadataCatalog.get(dataset_name).set(
        image_root=image_root,
        panoptic_root=panoptic_root,
        sem_seg_root=semantic_root,
        json_file=instances_json,
        panoptic_json=panoptic_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_coco_olac_panoptic(root):
    for (
            dataset_name,
            (image_root, panoptic_root, semantic_root, instance_json, panoptic_json,),
    ) in _PREDEFINED_SPLITS_COCO_OLAC.items():
        register_coco_olac_panoptic(
            dataset_name,
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, semantic_root),
            os.path.join(root, instance_json),
            os.path.join(root, panoptic_json),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_olac_panoptic(_root)
