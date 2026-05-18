import os
import json


ocl_dict = {}
im_root = '/home/wenbo/data/datasets/coco_ocl'
for dir in os.listdir(im_root):
    if dir in ['val_high', 'val_mid', 'val_low']:
        for im in os.listdir(os.path.join(im_root, dir, dir)):
            ocl_level = dir[4:]
            ocl_dict[im[:-4]] = ocl_level
occlusion_val_label = 'occlusion_label_val.json'
with open(occlusion_val_label, 'w') as ovl:
    json.dump(ocl_dict, ovl)
