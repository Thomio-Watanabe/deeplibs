from image_modules import images_dataset
import numpy as np
import struct
import os

'''
Load mnist dataset from: http://yann.lecun.com/exdb/mnist/
Thomio Watanabe 2016
'''


class MnistDataset():
    def __init__(self, num_rows = 28, num_cols = 28):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.image_size = num_rows
        self.num_labels = 10
        self.num_channels = 1
        self.train_size = 60000
        self.test_size = 10000

    def load_images( self, images_dir):
        print( '-- Loading mnist images...' )
        try:
            train_file = 'train-images.idx3-ubyte'
            path_to_train = os.path.join( images_dir, train_file )
            self.train_images = load_images_file( path_to_train , self.train_size, self.num_rows, self.num_cols )
        except IOError:
            train_file = 'train-images-idx3-ubyte'
            path_to_train = os.path.join( images_dir, train_file )
            self.train_images = load_images_file( path_to_train , self.train_size, self.num_rows, self.num_cols )
        try:
            test_file = 't10k-images.idx3-ubyte'
            path_to_test = os.path.join( images_dir, test_file )
            self.test_images = load_images_file( path_to_test, self.test_size, self.num_rows, self.num_cols )
        except IOError:
            test_file = 't10k-images-idx3-ubyte'
            path_to_test = os.path.join( images_dir, test_file )
            self.test_images = load_images_file( path_to_test, self.test_size, self.num_rows, self.num_cols )

    def load_labels( self, labels_dir ):
        print( '-- Loading mnist labels...' )
        try:
            train_file = 'train-labels.idx1-ubyte'
            path_to_train = os.path.join( labels_dir, train_file )
            self.train_labels = load_labels_file( path_to_train )
        except IOError:
            train_file = 'train-labels-idx1-ubyte'
            path_to_train = os.path.join( labels_dir, train_file )
            self.train_labels = load_labels_file( path_to_train )
        try:
            test_file = 't10k-labels.idx1-ubyte'
            path_to_test = os.path.join( labels_dir, test_file )
            self.test_labels = load_labels_file( path_to_test )
        except IOError:
            test_file = 't10k-labels-idx1-ubyte'
            path_to_test = os.path.join( labels_dir, test_file )
            self.test_labels = load_labels_file( path_to_test )

    def format_dataset( self,
                        normalize = True,
                        validation_size = 5000,
                        batch_size = 64,
                        num_epochs = 10,
                        eval_batch_size = 64,
                        eval_frequency = 100):
        self.train_images = self.train_images.reshape( self.train_size, self.image_size, self.image_size, self.num_channels).astype(np.float32)
        self.test_images = self.test_images.reshape( self.test_size, self.image_size, self.image_size, self.num_channels).astype(np.float32)
        if normalize:
            self.train_images = images_dataset.normalize_images( self.train_images )
            self.test_images = images_dataset.normalize_images( self.test_images )

        self.validation_data = self.train_images[:validation_size, ...]
        self.validation_labels = self.train_labels[:validation_size]
        self.train_data = self.train_images[validation_size:, ...]
        self.train_labels = self.train_labels[validation_size:]
        self.test_data = self.test_images
        self.test_labels = self.test_labels

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.eval_batch_size = eval_batch_size
        self.eval_frequency = eval_frequency



def load_images_file( path_to_file, number_images, num_rows, num_cols ):
    with open( path_to_file, 'rb' ) as image_file:
        magic, num, rows, cols = struct.unpack( ">IIII", image_file.read(16) )
        images = np.fromfile( image_file, dtype=np.uint8 ).reshape( number_images, num_rows, num_cols)
        images_dataset.analyse_images( images )
    return images


def load_labels_file( path_to_file ):
    with open( path_to_file, 'rb' ) as label_file:
        magic, num = struct.unpack( ">II", label_file.read(8) )
        labels = np.fromfile( label_file, dtype=np.int8 )
    return labels


