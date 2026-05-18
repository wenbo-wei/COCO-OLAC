cd "$(dirname "$0")/.." || exit 1
export DETECTRON2_DATASETS=datasets/data

python ./train_net.py \
--num-gpus 2 \
--config-file ./configs/coco_olac/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
OUTPUT_DIR ./output/coco_olac/res50/con \
SEED 801 \
DATALOADER.NUM_WORKERS 16 \
SOLVER.IMS_PER_BATCH 32 \
SOLVER.STEPS '42084, 44422' \
SOLVER.MAX_ITER 46760 \
INPUT.IMAGE_SIZE 512 \
CONTRASTIVE.ON True \
CONTRASTIVE.OCCLUSION_ANN datasets/data/coco_olac/occlusion_label_train.json \
CONTRASTIVE.OUTER_THRESHOLD 0.4 \
CONTRASTIVE.INNER_THRESHOLD_ON True \
CONTRASTIVE.INNER_THRESHOLD 0.6 \
INPUT.MIN_SIZE_TEST 400 \
INPUT.MAX_SIZE_TEST 666
