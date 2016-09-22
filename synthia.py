#!/usr/bin/env python
from image_modules import synthia_dataset
from tf_modules import udacity


dataset = synthia_dataset.SynthiaDataset()

training_dir = '../datasets/synthia_cvpr2016_small/RGB/'
dataset.load_images( training_dir )

gt_dir = '../datasets/synthia_cvpr2016_small/GTTXT/'
dataset.load_gt( gt_dir, object_name = 'Car')
dataset.create_gt_grid( [25,27] )

images_info, data_size, input_data = dataset.format_dataset()
udacity.model(images_info, data_size, input_data)