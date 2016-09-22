#!/usr/bin/env python
from image_modules import mnist_dataset
from tf_modules import udacity


dataset = mnist_dataset.MnistDataset()

images_dir = '../datasets/mnist'
dataset.load_images( images_dir )

labels_dir = '../datasets/mnist'
dataset.load_labels( labels_dir )

# images_info, data_size, input_data = dataset.format_dataset( model_type = 'classification')

