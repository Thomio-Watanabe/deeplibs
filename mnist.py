#!/usr/bin/env python
from image_modules import mnist_dataset
from tf_modules import LeNet


dataset = mnist_dataset.MnistDataset()

images_dir = '../datasets/mnist'
dataset.load_images( images_dir )

labels_dir = '../datasets/mnist'
dataset.load_labels( labels_dir )

dataset.format_dataset()

LeNet.model( dataset )
# AlexNet.model( dataset )
# Inception.model( dataset )
