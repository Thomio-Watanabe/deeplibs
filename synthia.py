#!/usr/bin/env python
from detection_modules import load_images
from detection_modules import load_labels
from detection_modules import bouding_box
from detection_modules import make_grid
from tf_modules import udacity
import numpy as np


from scipy import ndimage

training_dir = '../datasets/synthia_cvpr2016_small/RGB/'

# names, images = load_images.load_and_analyse( training_dir )
names, images = load_images.load( training_dir, nrows = 720, ncols = 960 )
# Transform list to array
images = np.array( images )

# Synthia dataset doesn't have a bb, each pixel is defined within a class
labels_dir = '../datasets/synthia_cvpr2016_small/GTTXT/'
# labels = load_labels.load( labels_dir, dataset = 'synthia', class = 'Car' )
