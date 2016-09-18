#!/usr/bin/env python
from image_modules import load_images
from image_modules import load_labels
from image_modules import bouding_box
from image_modules import make_grid
from tf_modules import udacity
import numpy as np



nrows = 375
ncols = 1242

training_dir = '../datasets/kitti_small/image_2'
names, images = load_images.load( training_dir, nrows, ncols )
# Transform list to array
images = np.array( images )

labels_dir = '../datasets/kitti_small/label_2'
labels = load_labels.load( labels_dir, 'Car' )

bb = bouding_box.get_bb_pixels(names, labels, nrows, ncols)
bb = np.array( bb )

grid = [25,27]
bb = make_grid.generate( bb, grid )

# ---------------------------------------------------------------------------------

# Transform each 2D image in an unidimentional array
nimages = len( images )
# npixels = nrows * ncols
num_channels = 1
# images = images.reshape(nimages, npixels)
images = images.reshape( (-1, nrows, ncols, num_channels)).astype(np.float32)

bb_nrows = bb.shape[1]
bb_ncols = bb.shape[2]
output_size = bb_nrows * bb_ncols
# bb = bb.reshape( nimages, output_size )
bb = bb.reshape( (-1, bb_nrows * bb_ncols)).astype(np.float32)
# 375 * 1242 = 465750 pixels each pixel is an input to the neural network


# Separate dataset in training, validation and test
# training = 60%
# validation and test = 20%
nimages_training = int( 0.6 * nimages )
nimages_validation = int( 0.2 * nimages )
nimages_test = int( 0.2 * nimages )

images_training = images[0:nimages_training, :]
# images_validation
# images_test

bb_training = bb[0:nimages_training, :]
# bb_validation
# bb_test


images_info = nrows, ncols, num_channels, output_size
data_size = nimages_training, nimages_validation, nimages_test
input_data = images_training, bb_training

udacity.model(images_info, data_size, input_data)
