cd "$(dirname "$0")/.." || exit 1
export DETECTRON2_DATASETS=datasets/data

python ./train_net.py \
--num-gpus 2 \
--config-file ./configs/coco_olac/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
--eval-only \
MODEL.WEIGHTS output/coco_olac/res50/con/model_final.pth \
OUTPUT_DIR ./output/coco_olac/res50/con_eval \
INPUT.IMAGE_SIZE 512 \
INPUT.MIN_SIZE_TEST 400 \
INPUT.MAX_SIZE_TEST 666
