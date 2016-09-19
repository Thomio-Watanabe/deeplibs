#!/usr/bin/env python
from image_modules import datasets


kitti_dataset = datasets.KittiDataset()
training_dir = '../datasets/kitti_small/test/'
kitti_dataset.load_images( training_dir )

gt_dir = '../datasets/kitti_small/test_label'
kitti_gt = kitti_dataset.load_gt( gt_dir, 'Car' )