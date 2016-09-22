from scipy import ndimage
import numpy as np
import abc
import os


# All datasets inherits from ImagesDataset
class ImagesDataset:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    def load_images( self, training_dir, analyse = False ):
        if( analyse ):
            self.names, self.images, self.nrows, self.ncols = choose_images( training_dir )
        else:
            self.names, self.images = load( training_dir, self.nrows, self.ncols )

    @abc.abstractmethod
    def load_gt(self):
        pass

    # divide the image in grid blocs
    def create_gt_grid( self, grid = [25,27] ):
        self.ground_truth = reduce_gt(grid, self.ground_truth)

    def format_dataset( self, model_type ):
        # classification models have labels instead of ground_truth
        model_classes = ['classification', 'segmentation', 'detection']
        if model_type not in model_classes:
            print( '-- Model type ', model_type,' not found.' )
            print( '-- Possible options are: ', model_classes )
            raise SystemExit

        return format_dataset( self.images, self.nrows, self.ncols, self.ground_truth )



def rgb2grey(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def normalize_image( array_2D ):
    pixel_depth = 255.0
    return ( array_2D - (pixel_depth) / 2 ) / pixel_depth


def analyse_images( images ):
    img = images[0]
    print( '-- Image array format:', img.shape )
    print( '-- Total number of images loaded: ', len( images ) )
    print( '-- Intensity max value:', np.amax(img) )
    print( '-- Intensity min value:', np.amin(img) )


'''
REWRITE this function to:
 - add zeros if the image is small than nrows x ncols
 - remove pixels if the image is larger than nrows x ncols
'''
def load( training_dir, nrows, ncols, grey_scale = True ):
    '''Load images with nrows and ncols.

    In some datasets the images doesn't have equal number of rows and columns.

    Args:
        training_dir (str): String with the training images directory.
        nrows (int): Number of rows of the images.
        ncols (int): Number of cols of the images.
        grey_scale (bool): Flag to transfor RGB images into grey scale.
    '''
    image_files = os.listdir( training_dir )
    # Index array has all the images with the parsed nrows and ncols
    names = []
    images = []
    print( '-- Loading (', nrows, 'x', ncols, ') images...' )
    for image_index, image_name in enumerate( image_files ):
        image_file = os.path.join( training_dir, image_name )
        image_array = ndimage.imread( image_file ).astype(float)
        if( (nrows == len(image_array)) and (ncols == len(image_array[0])) ):
            if( grey_scale ):
                image_array = rgb2grey( image_array )
            normalized_image = normalize_image( image_array )
            names.append( image_name )
            images.append( normalized_image )
    analyse_images( images )
    images = np.array(images)
    return names, images


def choose_images( training_dir, grey_scale = True ):
    '''Analyse the images from training_dir and load the images with the same nrows and ncols.

    In some datasets the images doesn't have equal number of rows and columns.
    This function load the separate the images in subsets based in their nrows and ncols.
    Then we pick the largest subset.

    Args:
        training_dir (str): String with the training images directory.
    '''
    names = []
    images = []
    rows = []
    cols = []
    image_files = os.listdir( training_dir )
    print( '-- Loading all images...' )
    for image_index, image_name in enumerate( image_files ):
        names.append( image_name )
        image_file = os.path.join( training_dir, image_name )
        image_array = ndimage.imread( image_file ).astype(float)
        rows.append( len(image_array) )
        cols.append( len(image_array[0]) )
        images.append( image_array )

    unique, counts = np.unique( rows, return_counts = True )
    index = np.argmax(counts)
    print( '-- Frequency of the number of rows: ' )
    print( np.asarray((unique, counts)).T )
    nrows = unique[index]

    unique, counts = np.unique( cols, return_counts = True )
    index = np.argmax(counts)
    print( '-- Frequency of the number of cols: ' )
    print( np.asarray((unique, counts)).T )
    ncols = unique[index]

    # Remove images with different number of nrows x ncols
    for i in reversed( range(len(images)) ):
        if( nrows != len(images[i]) or ncols != len(images[i][0]) ):
            images.pop(i)
            names.pop(i)

    print( '-- Number of rows more frequent:', nrows )
    print( '-- Number of cols more frequent:', ncols )

    for i in range( len(images) ):
        if( grey_scale ):
            images[i] = rgb2grey( images[i] )
        images[i] = normalize_image( images[i] )

    analyse_images( images )
    images = np.array(images)
    return names, images, nrows, ncols


def reduce_gt(grid, ground_truth):
    print( '-- Creating grid for the ground truth...' )
    n_bb = ground_truth.shape[0]
    horizontal_size = int(ground_truth.shape[1] / grid[0])
    vertical_size = int(ground_truth.shape[2] / grid[1])
    reduced_ground_truth = np.ndarray( (n_bb , horizontal_size, vertical_size), float)
    reduced_ground_truth.fill(0)
    for k in range( n_bb ):
        for i in range ( int(horizontal_size) ):
            row = i * grid[0]
            for j in range ( int(vertical_size) ):
                column = j * grid[1]
                box = ground_truth[k, row:(row + grid[0]), column:(column + grid[1]) ]
                n_ones = np.sum( box.reshape(1, box.size) == 1 )
                percentage = np.divide( n_ones, grid[0] * grid[1] ,dtype = float )
                # if more the 90% f the box have ones => the area will keep the value
                if ( percentage >= 0.9 ):
                    reduced_ground_truth[k, i, j] = 1
    return reduced_ground_truth


def format_dataset( images, nrows, ncols, ground_truth ):
    # Transform each 2D image in an unidimentional array
    nimages = len( images )
    num_channels = 1
    images = images.reshape( (-1, nrows, ncols, num_channels)).astype(np.float32)

    gt_nrows = ground_truth.shape[1]
    gt_ncols = ground_truth.shape[2]
    output_size = gt_nrows * gt_ncols
    ground_truth = ground_truth.reshape( (-1, gt_nrows * gt_ncols)).astype(np.float32)

    # Separate dataset in training, validation and test
    # training = 60%
    # validation and test = 20%
    nimages_training = int( 0.6 * nimages )
    nimages_validation = int( 0.2 * nimages )
    nimages_test = int( 0.2 * nimages )

    images_training = images[0:nimages_training, :]
    # images_validation
    # images_test

    ground_truth_training = ground_truth[0:nimages_training, :]
    # bb_validation
    # bb_test

    images_info = nrows, ncols, num_channels, output_size
    data_size = nimages_training, nimages_validation, nimages_test
    input_data = images_training, ground_truth_training

    return images_info, data_size, input_data
