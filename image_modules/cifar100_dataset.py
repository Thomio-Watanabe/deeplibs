'''
Dataset obtained from:
https://www.cs.toronto.edu/~kriz/cifar.html
'''
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
        '''load_images

        Each unpickled batch file is a dictionary containing: dict_keys(['filenames', 'batch_label', 'labels', 'data'])

        Images are store as 10000x3072 numpy array of uint8s.
        Each row of the array stores a 32x32 colour image.
        The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
        The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
        '''
        print('-- Loading images...')
        train_file = os.path.join( images_dir, train_file_name )
        test_file = os.path.join( images_dir, test_file_name )

        train_batch = unpickle( train_file )
        test_batch = unpickle( test_file )

        # Cifar100 has 100 classes (fine_labels) separated in 20 superclasses (coarse_labels)
        class_separation = 'fine_labels', 'coarse_labels'
        self.train_images, self.train_labels = extract_data( train_batch, 50000, class_separation[1] )
        self.test_images, self.test_labels = extract_data( test_batch, 10000, class_separation[1] )

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
        print('-- Formating dataset...')

        if grey_scale:
            self.train_images = images_dataset.rgb2grey( self.train_images )
            self.test_images = images_dataset.rgb2grey( self.test_images )
            self.num_channels = 1

        train_size = self.train_images.shape[0]
        test_size = self.test_images.shape[0]
        self.train_images = self.train_images.reshape( train_size, self.num_rows, self.num_cols, self.num_channels).astype(np.float32)
        self.test_images = self.test_images.reshape( test_size, self.num_rows, self.num_cols, self.num_channels).astype(np.float32)
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
    '''
    batch['data'] is a numpy array of 10000 imagels and 3072 pixels/image
    batch['labels'] is a list 10000 labels (from 0 to 9)
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

    # print('Batch keys:', batch.keys() )
    labels = np.asarray( batch[class_separation] )
    return reshaped_images, labels

