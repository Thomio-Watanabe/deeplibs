#!/usr/bin/env python
from image_modules import datasets


synthia_dataset = datasets.SynthiaDataset()
training_dir = '../datasets/synthia_cvpr2016_small/test/'
synthia_dataset.load_images( training_dir )

gt_dir = '../datasets/synthia_cvpr2016_small/test_GTTXT/'
gt = synthia_dataset.load_gt( gt_dir, object_name = 'Car')