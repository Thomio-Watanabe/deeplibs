from image_modules import images_dataset
from scipy import ndimage
import numpy as np
import tarfile
import random
import os


class ILSVRC2012Dataset( images_dataset.ImagesDataset ):
    def __init__( self, nrows = 256, ncols = 256 ):
        print('-------------------------')
        print('-- ILSVRC 2012 Dataset --')
        print('-------------------------')
        self.num_rows = nrows
        self.num_cols = ncols
        self.num_labels = 1000
        self.num_channels = 3

    def load_val_labels( self, gt_dir ):
        self.val_labels = load_labels( gt_dir )

    def load_images( self, training_dir, num_labels=1000 ):
        self.num_labels = num_labels
        self.names, self.images_list, self.image_labels = load_ilsvrc2012_images( training_dir, num_labels )
        self.names, self.images, self.images_index = images_dataset.resize_images( self.names, self.images_list, self.num_rows, self.num_cols, self.num_channels )

    def format( self,
                gray_scale = False,
                normalize = True,
                batch_size = 128,
                num_epochs = 10,
                eval_batch_size = 128,
                eval_frequency = 10):
        if gray_scale:
            self.images = images_dataset.rgb2gray( self.images )
            self.num_channels = 1

        if normalize:
            self.images = images_dataset.normalize_images( self.images )

        # Format labels as 1D array of 1 element
#        self.val_labels = self.val_labels.ravel()
        self.image_labels = self.image_labels.ravel()
        # Get labels correspondent to each image
#        self.val_labels = self.val_labels[ self.images_index ]
        self.image_labels = self.image_labels[ self.images_index ]

        # Format images as 1D array of 4 elements
        _num_images = len( self.images_index )
        self.images = self.images.reshape( _num_images, self.num_rows, self.num_cols, self.num_channels ).astype(np.float32)

        # Shuffle images and labels
        _random_index = np.arange( _num_images )
        random.shuffle( _random_index )
        self.images = self.images[ _random_index ]
        self.image_labels = self.image_labels[ _random_index ]

        # 20% os the images and labels go for validation and test
        validation_size = int( _num_images / 10 )
        test_size = int( _num_images / 10 )

        self.validation_data = self.images[:validation_size, ...]
        self.validation_labels = self.image_labels[:validation_size]
#        self.validation_labels = self.val_labels[:validation_size]
        self.train_data = self.images[validation_size:, ...]
#        self.train_labels = self.val_labels[validation_size:]
        self.train_labels = self.image_labels[validation_size:]

        self.test_data = self.train_data[:test_size, ...]
        self.test_labels = self.train_labels[:test_size]
        self.train_data = self.train_data[test_size:, ...]
        self.train_labels = self.train_labels[test_size:]

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.eval_batch_size = eval_batch_size
        self.eval_frequency = eval_frequency

        images_dataset.images_info( self.images )


def load_val_labels( gt_dir ):
    gt_name = 'ILSVRC2012_validation_ground_truth.txt'
    gt_file = os.path.join( gt_dir, gt_name)
    with open( gt_file ) as fileHandle:
        array = [[int(x) for x in line.split()] for line in fileHandle]
    ground_truth = np.array( array )
    return ground_truth


def load_ilsvrc2012_images( training_dir, num_labels ):
    '''Load ILSVRC2012 images from .tar files

    This function will load ILSVRC2012 images.
    The images MUST be compressed in .tar
    Each .tar file represents one dataset class (label).

    Args:
        training_dir (str): String with path to .
        num_labels (int): Number of classes (labels) that should be loaded.

    Returns:
        names (list[str]): List with all images names.
        images (list[ndarray]): List with images (multidimensional array).
        image_labels (list[int]): List with each loaded image label.
    '''
    print('-- Loading images...')
    # Load all tarfile names from training_dir
    tarfile_name = os.listdir( training_dir )

    # Load ILSVRC2012 class names
    fh = open('./image_modules/ilsvrc2012_class_names.txt')
    classes = fh.readlines()
    # Remove new line character
    for i in range(len(classes)):
        classes[i] = classes[i].strip('\n')

    names = []
    images = []
    image_labels = []
    # Load class 1 to num_labels
    for i in range( num_labels ):
        print('Extracting', tarfile_name[i] )
        tarfile_path = os.path.join( training_dir, tarfile_name[i] )
        tarfile_handle = tarfile.open( tarfile_path )
        file_names = tarfile_handle.getnames()
        file_members = tarfile_handle.getmembers()
        # Extract images from each label (class)
        for j in range(len(file_members)):
            image = tarfile_handle.extractfile( file_members[j] )
            image_array = ndimage.imread( image ).astype(float)
            images.append( image_array )
            names.append( file_names[j] )
            # The first i is 0 -> makes 1 the first label
            image_labels.append( i + 1 )

    image_labels = np.array( image_labels )
    print('Total number of images loaded:', len(images) )
    return names, images, image_labels

