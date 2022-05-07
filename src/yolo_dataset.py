# Copyright 2021 PCL
# ============================================================================
"""Pulsar dataset."""
import os
import multiprocessing
from PIL import Image
import cv2

import numpy as np
import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as CV

from src.distributed_sampler import DistributedSampler
from src.transforms import reshape_fn, Transformer, MultiScaleTrans
import json

class PulsarYoloDataset():
    """
    Pulsar yolo dataset definitation.
    """
    def __init__(self, root, data_txt, cfg, is_training):
        self.root = root
        image_list = os.listdir(self.root)
        self.anns = json.load(open(data_txt, 'r'))
        self.img_path = image_list
        self.cfg = cfg
        self.is_training = is_training
        self.transformer = Transformer(
                                     scale_tsamp=cfg['scale_tsamp'],
                                     fch1=cfg['fch1'],
                                     fend=cfg['fend'],
                                     padding=1
                                    )

    def __getitem__(self, index):
        img_path = self.img_path
        filname = img_path[index]
        img = Image.open(os.path.join(self.root, filname)).convert("L")
        ann_dict = self.anns[filname[:-4]]
        isfake = filname[:4] == 'fake'
        ann = []
        for a in ann_dict:
            ann.append([a["time"],a["DM"], a['start_freq'], a['end_freq'], isfake])
        ann = np.array(ann, dtype=np.float)
        ann = self.transformer.ann_trans(ann, self.cfg['num_samps'])
        bbox = []
        for ind in range(ann.shape[0]):
            bbox.append(list(ann[ind].astype('float16')))
        if self.is_training:
            return img, bbox, [], [], [], [], [], []
        else:
            # return filname, img, bbox
            name_split = filname.split('.png')[0].split('_')
            img_id = np.array(name_split[1:], dtype=np.float32)
            return img, bbox, img_id

    def __len__(self):
        return len(self.img_path)

def create_yolo_dataset(image_dir,
                          data_txt,
                          batch_size,
                          max_epoch,
                          device_num,
                          rank,
                          config=None,
                          is_training=True):
    """
    Create yolo dataset.
    """
    yolo_dataset = PulsarYoloDataset(root = image_dir, data_txt = data_txt, cfg = config.data_pre, is_training = is_training)
    distributed_sampler = DistributedSampler(len(yolo_dataset), device_num, rank, shuffle=is_training)

    config.dataset_size = len(yolo_dataset)
    cores = multiprocessing.cpu_count()
    num_parallel_workers = int(cores / device_num)
    if is_training:
        dataset_column_names = ["img", "ann", "bbox1", "bbox2", "bbox3",
                                        "gt_box1", "gt_box2", "gt_box3"]
        multi_scale_trans = MultiScaleTrans(config, device_num)
        if device_num != 8:
            ds = de.GeneratorDataset(yolo_dataset, column_names=dataset_column_names,
                                         num_parallel_workers=min(32, num_parallel_workers),
                                         sampler=distributed_sampler)
            ds = ds.batch(batch_size, per_batch_map=multi_scale_trans, input_columns=dataset_column_names,
                              num_parallel_workers=min(32, num_parallel_workers), drop_remainder=True)
        else:
            ds = de.GeneratorDataset(yolo_dataset, column_names=dataset_column_names,sampler=distributed_sampler)
            ds = ds.batch(batch_size, per_batch_map=multi_scale_trans, input_columns=dataset_column_names,
                              num_parallel_workers=min(8, num_parallel_workers), drop_remainder=True)

    else:
        ds = de.GeneratorDataset(yolo_dataset, column_names=["img", "ann", "img_id"],
                                  sampler=distributed_sampler)
    
        compose_map_func = (lambda img, ann, img_id: reshape_fn(img, ann, config, img_id))
        ds = ds.map(input_columns=["img", "ann", "img_id"],
                    output_columns=["img", "ann", "img_id"],
                    column_order=["img", "ann", "img_id"],
                    operations=compose_map_func, num_parallel_workers=8)
        ds = ds.batch(batch_size, drop_remainder=is_training)

    ds = ds.repeat(max_epoch)
    return ds, len(yolo_dataset)
