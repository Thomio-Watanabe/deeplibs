#!/usr/bin/env python
from image_modules import kitti_dataset


dataset = kitti_dataset.KittiDataset()

training_dir = '../datasets/kitti_small/test/'
dataset.load_images( training_dir )

gt_dir = '../datasets/kitti_small/test_label'
gt = dataset.load_gt( gt_dir, 'Car' )