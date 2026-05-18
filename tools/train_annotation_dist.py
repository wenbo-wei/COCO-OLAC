import json


thing_cls = []
ocl_train = '/home/wenbo/data/datasets/coco_ocl/occlusion_label_train_30k.json'
with open(ocl_train, 'r') as f:
    ocl_train_dic = json.load(f)

train_30k_ann = '/home/wenbo/data/datasets/coco_ocl/train_30k/annotations/panoptic_train_30k.json'
with open(train_30k_ann, 'r') as f:
    train_30k_dic = json.load(f)

for cls in train_30k_dic['categories']:
    if cls['isthing'] == 1:
        thing_cls.append(cls['id'])

high_ins = 0
mid_ins = 0
low_ins = 0

for ann in train_30k_dic['annotations']:
    for seg in ann['segments_info']:
        if seg['category_id'] in thing_cls and seg['iscrowd'] != 1:
            if ocl_train_dic[ann['file_name'][:-4]] == 'high':
                high_ins += 1
            if ocl_train_dic[ann['file_name'][:-4]] == 'mid':
                mid_ins += 1
            if ocl_train_dic[ann['file_name'][:-4]] == 'low':
                low_ins += 1
            else:
                break
ins_per_im_high = high_ins / 12081
ins_per_im_mid = mid_ins / 11251
ins_per_im_low = low_ins / 6668

ins_val_high = '/home/wenbo/data/datasets/coco_ocl/val_high/annotations/panoptic_val_high.json'
ins_val_mid = '/home/wenbo/data/datasets/coco_ocl/val_mid/annotations/panoptic_val_mid.json'
ins_val_low = '/home/wenbo/data/datasets/coco_ocl/val_low/annotations/panoptic_val_low.json'

with open(ins_val_high, 'r') as f:
    high = json.load(f)

high_ins = 0

for cls in high['categories']:
    if cls['isthing'] == 1:
        thing_cls.append(cls['id'])

for ann in high['annotations']:
    for seg in ann['segments_info']:
        if seg['category_id'] in thing_cls and seg['iscrowd'] != 1:
            high_ins += 1
ins_per_im_high = high_ins / len(high['annotations'])

with open(ins_val_mid, 'r') as f:
    mid = json.load(f)
    mid_ins = 0
for ann in mid['annotations']:
    for seg in ann['segments_info']:
        if seg['category_id'] in thing_cls and seg['iscrowd'] != 1:
            mid_ins += 1
ins_per_im_mid = mid_ins / len(mid['annotations'])

with open(ins_val_low, 'r') as f:
    low = json.load(f)
low_ins = 0
for ann in low['annotations']:
    for seg in ann['segments_info']:
        if seg['category_id'] in thing_cls and seg['iscrowd'] != 1:
            low_ins += 1
ins_per_im_low = low_ins / len(low['annotations'])

print('')