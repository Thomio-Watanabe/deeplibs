#!/usr/bin/env python
from image_modules import kitti_dataset
from tf_modules import udacity


dataset = kitti_dataset.KittiDataset()

training_dir = '../datasets/kitti_small/test/'
dataset.load_images( training_dir )

gt_dir = '../datasets/kitti_small/test_label'
dataset.load_gt( gt_dir, object_name = 'Car' )
dataset.create_gt_grid( [25,27] )

images_info, data_size, input_data = dataset.format_dataset()
udacity.model(images_info, data_size, input_data)