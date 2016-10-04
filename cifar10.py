#!/usr/bin/env python
from image_modules import cifar10_dataset
from tf_modules import LeNet


dataset = cifar10_dataset.Cifar10Dataset()

images_dir = '../datasets/cifar10'
dataset.load_images( images_dir )

dataset.format_dataset()

LeNet.model( dataset )
