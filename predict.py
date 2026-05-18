import os
import cv2
from PIL import Image

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

# import Mask2Former project
from mask2former import add_maskformer2_config
from contrastive import add_contrastive_config


class Predictor:
    def setup(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_contrastive_config(cfg)
        cfg.merge_from_file("./configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml")
        # cfg.MODEL.WEIGHTS = './model_zoo/coco_r50.pkl'
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True

        # cfg.MODEL.WEIGHTS = '/home/wenbo/data/projects/Mask2Former_ConOcc/outputs/coco_30k_size512/res50/ref_size800_1333/con_3cl_out0.4_in0.6_40.1/model_final.pth'
        # cfg.CONTRASTIVE.ON = True
        # cfg.MODEL.WEIGHTS = '/home/wenbo/data/projects/Mask2Former_ConOcc/outputs/coco_30k_size512/res50/ref_size800_1333/base_50ep_39.7/model_final.pth'
        # cfg.CONTRASTIVE.ON = False
        cfg.MODEL.WEIGHTS = '/home/wenbo/data/projects/Mask2Former_ConOcc/outputs/model_final_94dc52.pkl'
        cfg.CONTRASTIVE.ON = False

        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        self.coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

    def predict(self, image):
        img_name = os.path.splitext(os.path.basename(image))[0]
        im = cv2.imread(str(image))
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"),
                                              outputs["panoptic_seg"][1]).get_image()
        # v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        # instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        # v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        # semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
        # result = np.concatenate((panoptic_result, instance_result, semantic_result), axis=0)[:, :, ::-1]

        out_name = f"{img_name}.pdf"
        out_path = os.path.join(".", out_name)
        img = Image.fromarray(panoptic_result)
        img.save(out_path, "PDF")
        return out_path


if __name__ == "__main__":
    image_predictor = Predictor()
    im_dir = '/home/wenbo/data/datasets/coco_olac/val/val/000000036678.jpg'
    image_predictor.setup()
    image_predictor.predict(im_dir)