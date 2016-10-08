#!/usr/bin/env python
from image_modules.mnist_dataset import MnistDataset
from tf_modules.lenet import LeNet
from tf_modules.alexnet import AlexNet


dataset = MnistDataset()

images_dir = '../datasets/mnist'
dataset.load_images( images_dir )

labels_dir = '../datasets/mnist'
dataset.load_labels( labels_dir )

dataset.format_dataset()

model = LeNet( dataset )
# model = AlexNet( dataset )
# model = GoogleNet( dataset )

model.train()
