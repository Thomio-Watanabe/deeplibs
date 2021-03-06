'''
Dataset obtained from:
https://www.cs.toronto.edu/~kriz/cifar.html

https://code.google.com/p/cuda-convnet/
'''
from image_modules import images_dataset
import numpy as np
import pickle
import os



class Cifar10Dataset():
    def __init__( self, num_rows = 32, num_cols = 32 ):
        self.num_rows = 32
        self.num_cols = 32
        self.num_labels = 10
        self.num_channels = 3

    def load_images( self,
                     images_dir,
                     file_name_01 = 'data_batch_1',
                     file_name_02 = 'data_batch_2',
                     file_name_03 = 'data_batch_3',
                     file_name_04 = 'data_batch_4',
                     file_name_05 = 'data_batch_5',
                     file_name_06 = 'test_batch' ):
        '''Load images from batch files.

        Each unpickled batch file is a dictionary containing: dict_keys(['filenames', 'batch_label', 'labels', 'data'])

        Images are store as 10000x3072 numpy array of uint8s.
        Each row of the array stores a 32x32 colour image.
        The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
        The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

        Args:
            images_dir (str): Directory with the training and test images files.
            file_name_0X (str): File name with the training/testing images.
        '''
        print('-------------')
        print('-- Cifar10 --')
        print('-------------')

        print('-- Loading images...')
        batch_file_01 = os.path.join( images_dir, file_name_01 )
        batch_file_02 = os.path.join( images_dir, file_name_02 )
        batch_file_03 = os.path.join( images_dir, file_name_03 )
        batch_file_04 = os.path.join( images_dir, file_name_04 )
        batch_file_05 = os.path.join( images_dir, file_name_05 )
        test_file = os.path.join( images_dir, file_name_06 )

        batch_01 = unpickle( batch_file_01 )
        batch_02 = unpickle( batch_file_02 )
        batch_03 = unpickle( batch_file_03 )
        batch_04 = unpickle( batch_file_04 )
        batch_05 = unpickle( batch_file_05 )
        test_batch = unpickle( test_file )

        self.images_01, self.labels_01 = extract_data( batch_01 )
        self.images_02, self.labels_02 = extract_data( batch_02 )
        self.images_03, self.labels_03 = extract_data( batch_03 )
        self.images_04, self.labels_04 = extract_data( batch_04 )
        self.images_05, self.labels_05 = extract_data( batch_05 )
        self.images_06, self.labels_06 = extract_data( test_batch )

    def load_labels( self, labels_dir ):
        print('-- Labels were loaded with images.')

    def format_dataset( self,
                        grey_scale = True,
                        normalize = True,
                        validation_size = 5000,
                        batch_size = 64,
                        num_epochs = 10,
                        eval_batch_size = 64,
                        eval_frequency = 100 ):
        '''Format training and testing images.

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
        self.train_images = np.concatenate( (self.images_01,
                                             self.images_02,
                                             self.images_03,
                                             self.images_04,
                                             self.images_05),
                                             axis = 0 )
        self.test_images = self.images_06
        self.train_labels = np.concatenate( (self.labels_01,
                                             self.labels_02,
                                             self.labels_03,
                                             self.labels_04,
                                             self.labels_05),
                                             axis = 0 )
        self.test_labels = self.labels_06

        if grey_scale:
            self.train_images = images_dataset.rgb2grey_array( self.train_images )
            self.test_images = images_dataset.rgb2grey_array( self.test_images )
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

    Each unpickled batch file is a dictionary containing: dict_keys(['filenames', 'batch_label', 'labels', 'data'])

    Args:
        file_name (str): Path to file to unpickle.

    Returns:
        unpickled_data (dict of str: *): Dictionary with strings as keys: ['filenames', 'batch_label', 'labels', 'data'].
    '''
    file_handle = open(file_name, 'rb')
    try:    # python3
        unpickled_data = pickle.load(file_handle, encoding = 'latin1')
    except TypeError:   # python2
        unpickled_data = pickle.load(file_handle)
    return unpickled_data


def extract_data( batch,
                  num_images_batch = 10000,
                  num_rows = 32,
                  num_cols = 32,
                  num_channels = 3):
    '''Extract images and labels from dictionary object.

    batch['data'] is a numpy array of 10000 imagels and 3072 pixels/image
    batch['labels'] is a list 10000 labels (from 0 to 9)

    Args:
        batch (dict of str: *): Dictionary with strings as keys.
        num_images_batch (int): Number of images per batch.
        num_rows (int): Number of rows in each image.
        num_cols (int): number of cols in each image.
        num_channels (int): Number of color channels in each image.

    Returns:
        reshaped_images (ndarray): Multidimensional numpy array of images.
        labels (ndarray[int]): Numpy array of labels (integers).
    '''
    # reshape batch['data'] to 1000x3x1024 array
    pixels_channel = 32 * 32
    images = batch['data'].reshape( num_images_batch, num_channels, pixels_channel )

    # reshape batch['data'] to 1000x32x32x3 array
    reshaped_images = np.ndarray( shape=(num_images_batch, num_rows, num_cols, num_channels), dtype=np.uint8 )
    reshaped_images.fill(0)
    for i in range( num_images_batch ):
        image = images[i].transpose()
        reshaped_images[i] = image.reshape( num_rows, num_cols, num_channels )

    labels = np.asarray( batch['labels'] )
    return reshaped_images, labels


