'''
Load mnist dataset from: http://yann.lecun.com/exdb/mnist/
'''
from image_modules import images_dataset
import numpy as np
import struct
import os


class MnistDataset( images_dataset.ImagesDataset ):
    def __init__(self, nrows = 28, ncols = 28):
        self.nrows = nrows
        self.ncols = ncols

    # Overriding DatasetBase method
    def load_images( self,
                     images_dir, 
                     train_file = 'train-images-idx3-ubyte',
                     test_file = 't10k-images-idx3-ubyte' ):
        print( '-- Loading mnist images...')
        path_to_train = os.path.join( images_dir, train_file )
        path_to_test = os.path.join( images_dir, test_file )        
        self.train_images = load_images_file( path_to_train , 60000, self.nrows, self.ncols )
        self.test_images = load_images_file( path_to_test, 10000, self.nrows, self.ncols )

    def load_labels( self,
                           labels_dir,
                           train_file = 'train-labels-idx1-ubyte',
                           test_file = 't10k-labels-idx1-ubyte' ):
        print( '-- Loading mnist labels...')
        path_to_train = os.path.join( labels_dir, train_file )
        path_to_test = os.path.join( labels_dir, test_file ) 
        self.train_labels = load_labels_file( path_to_train )
        self.test_labels = load_labels_file( path_to_test )



def load_images_file( path_to_file, number_images, nrows, ncols ):
    with open( path_to_file, 'rb' ) as image_file:
        magic, num, rows, cols = struct.unpack( ">IIII", image_file.read(16) )
        images = np.fromfile( image_file, dtype=np.uint8 ).reshape( number_images, nrows, ncols)
        images_dataset.analyse_images( images )
    return images


def load_labels_file( path_to_file ):
    with open( path_to_file, 'rb' ) as label_file:
        magic, num = struct.unpack( ">II", label_file.read(8) )
        labels = np.fromfile( label_file, dtype=np.int8 )
    return labels

