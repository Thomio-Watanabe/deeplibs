'''
Dataset obtained from:
https://www.cs.toronto.edu/~kriz/cifar.html
'''
from __future__ import print_function
from image_modules import images_dataset
import numpy as np
import pickle
import os



class Cifar100Dataset():
    def __init__( self, num_rows = 32, num_cols = 32 ):
        self.num_rows = 32
        self.num_cols = 32
        self.num_labels = 100
        self.num_channels = 3

    def load_images( self,
                     images_dir,
                     train_file_name = 'train',
                     test_file_name = 'test' ):
        '''Load 'train' and 'test' files with images from 'images_dir'.

        Images are stored as 50000x3072 (train) and 10000x3072 (test) numpy array of uint8s.
        Each row of the array stores a 32x32 colour image.
        The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
        The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

        Args:
            images_dir (str): String with the training and test images directory.
            train_file_name (str): File name with the training images.
            test_file_name (str): File name with the test images.
        '''
        print('--------------')
        print('-- Cifar100 --')
        print('--------------')

        print('-- Loading images...')
        train_file = os.path.join( images_dir, train_file_name )
        test_file = os.path.join( images_dir, test_file_name )

        train_batch = unpickle( train_file )
        test_batch = unpickle( test_file )

        # Cifar100 has 100 classes (fine_labels) separated in 20 superclasses (coarse_labels)
        __class_separation = 'fine_labels', 'coarse_labels'
        __choice = 1
        print('-- Separating images in:', __class_separation[ __choice ])
        self.train_images, self.train_labels = extract_data( train_batch, 50000, __class_separation[ __choice ] )
        self.test_images, self.test_labels = extract_data( test_batch, 10000, __class_separation[ __choice ] )

    def load_labels( self, labels_dir ):
        print('-- Labels were loaded with images.')

    def format_dataset( self,
                        grey_scale = False,
                        normalize = True,
                        validation_size = 5000,
                        batch_size = 64,
                        num_epochs = 10,
                        eval_batch_size = 64,
                        eval_frequency = 100 ):
        '''Format dataset training and testing images.

        Args:
            grey_scale (bool): Flag to transform RGB images to grey scale.
            normalize (bool): Flag to normalize images (highly recommended).
            validation_size (int): Number of images in validation set.
            batch_size (int): Number of images in training batch.
            num_epochs (int): Number of times the training set is used.
            eval_batch_size (int): Number of images in the evaluation batch.
            eval_frequency (int): Number of steps required to evaluate the training.
        '''
        print('-- Formating dataset...')

        if grey_scale:
            self.train_images = images_dataset.rgb2grey( self.train_images )
            self.test_images = images_dataset.rgb2grey( self.test_images )
            self.num_channels = 1

        __train_size = self.train_images.shape[0]
        __test_size = self.test_images.shape[0]
        self.train_images = self.train_images.reshape( __train_size, self.num_rows, self.num_cols, self.num_channels).astype(np.float32)
        self.test_images = self.test_images.reshape( __test_size, self.num_rows, self.num_cols, self.num_channels).astype(np.float32)
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



def unpickle(file_name):
    '''Unpickle dataset files with latin1 encoding.

    Each unpickled batch file is a dictionary containing: dict_keys(['data', 'batch_label', 'fine_labels', 'coarse_labels', 'filenames'])

    Args:
        file_name (str): Path to file to unpickle.

    Returns:
        unpickled_data (dict of str: *): Dictionary with strings as keys: ['data', 'batch_label', 'fine_labels', 'coarse_labels', 'filenames'].
    '''
    file_handle = open(file_name, 'rb')
    try:    # python3
        unpickled_data = pickle.load(file_handle, encoding = 'latin1')
    except TypeError:   # python2
        unpickled_data = pickle.load(file_handle)
    return unpickled_data


def extract_data( batch,
                  num_images_batch,
                  class_separation,
                  num_rows = 32,
                  num_cols = 32,
                  num_channels = 3):
    '''Extract images and labels from dictionary object.

    batch['data'] is a numpy array of x images and 3072 pixels/image.
    batch['fine_labels'] is a list x labels (from 0 to 99).
    batch['coarse_labels'] is a list x labels (from 0 to 19). 20 superclasses of fine_labels.

    Args:
        batch (dict of str: *): Dictionary with strings as keys.
        num_images_batch (int): Number of images per batch.
        class_separation (str): Key to retrive labels from file. There are 2 options: 'fine_labels', 'coarse_labels'.
        num_rows (int): Number of rows in each image.
        num_cols (int): number of cols in each image.
        num_channels (int): Number of color channels in each image.

    Returns:
        reshaped_images (ndarray): Multidimensional numpy array of images.
        labels (ndarray[int]): Numpy array of labels (integers).
    '''
    # reshape batch['data'] to 50000x3x1024 or 10000x3x1024 array
    pixels_channel = 32 * 32
    images = batch['data'].reshape( num_images_batch, num_channels, pixels_channel )

    # reshape batch['data'] to 50000x32x32x3 or 10000x32x32x3 array
    reshaped_images = np.ndarray( shape=(num_images_batch, num_rows, num_cols, num_channels), dtype=np.uint8 )
    reshaped_images.fill(0)
    for i in range( num_images_batch ):
        image = images[i].transpose()
        reshaped_images[i] = image.reshape( num_rows, num_cols, num_channels )

    # print('Batch keys:', batch.keys() )
    labels = np.asarray( batch[class_separation] )
    return reshaped_images, labels

