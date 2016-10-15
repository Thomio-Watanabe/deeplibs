from __future__ import print_function
from scipy.misc import imresize
from scipy import ndimage
import numpy as np
import os


# All datasets inherits from ImagesDataset
class ImagesDataset:
    def __init__(self):
        pass

    def load_images( self, training_dir, choose = False, resize = False ):
        self.names, self.images_list, self.rows_list, self.cols_list = load_all_images( training_dir )

        if choose: # Analyse images and save those with more frequent nrows,ncols
            self.names, self.images, self.num_rows, self.num_cols, self.images_index = choose_images( self.names, self.images_list, self.rows_list, self.cols_list, self.num_channels)
        elif resize: # Load images and resize them with default nrows,ncols
            self.names, self.images, self.images_index = resize_images( self.names, self.images_list, self.num_rows, self.num_cols, self.num_channels )
        else: # Load images with given num_rows, num_cols (defined in their child class constructor)
            self.names, self.images, self.images_index = save_default( self.names, self.images_list, self.num_rows, self.num_cols, self.num_channels )

        # Print general info about the loaded images
        images_info( self.images )

    # divide the image in grid blocs
    def create_gt_grid( self, grid = [25,27] ):
        self.ground_truth = reduce_gt(grid, self.ground_truth)

    def format_dataset( self, gray_scale = False, normalize = True ):
        # Classification models have labels instead of ground_truth
        # model_classes = ['classification', 'segmentation', 'detection']
        # if self.model_type not in model_classes:
        #     print( '-- Model type ', model_type,' not found.' )
        #     print( '-- Possible options are: ', model_classes )
        #     raise SystemExit
        if gray_scale:
            self.images = rgb2gray( self.images )
            self.num_channels = 1

        if normalize:
            self.images = normalize_images( self.images )

        return format_dataset( self.images, self.num_rows, self.num_cols, self.ground_truth )
        # gray_scale and normalize functions dont change the number of loaded images



def rgb2gray( images_array ):
    '''Function to transform an array of RGB images to an array of gray scale images.

    Args:
        images_array (numpy array): numpy array with 3 elements (num_images, num_rows, num_cols).

    Returns:
        grey_images (numpy array): float32 numpy array.
    '''
    print('-- Transforming to gray scale...')
    print('Input images shape:', images_array.shape )
    num_images, num_rows, num_cols, num_channels = images_array.shape
    grey_images = np.ndarray( shape = (num_images, num_rows, num_cols, 1), dtype = np.float32 )
    for i in range( num_images ):
        img = images_array[i]
        grey_images[i] = np.dot( img[...,:3], [0.299, 0.587, 0.114] ).reshape(num_rows, num_cols, 1)
    return grey_images


def normalize_images( images_array, pixel_depth = 255.0 ):
    '''Function to normalize an array of images.

    Args:
        images_array (numpy array): numpy array with 3 elements (num_images, num_rows, num_cols).
        pixel_depth (Optional[float]: max values of pixel intensity. Default = 255.0.

    Returns:
        normalized_images (numpy array): float32 numpy array.
    '''
    print('-- Normalizing images...')
    print('Input images type:', images_array.dtype )
    print('Input images shape:', images_array.shape )
    num_images, num_rows, num_cols, num_channels = images_array.shape
    normalized_images = np.ndarray( shape = (num_images, num_rows, num_cols, num_channels), dtype = np.float32 )
    for i in range( num_images ):
        normalized_images[i] =  ( images_array[i] - (pixel_depth) / 2.0 ) / pixel_depth
    return normalized_images


def images_info( images ):
    img = images[0]
    print( '-- Analyzing images' )
    print( 'Image array format:', img.shape )
    print( 'Total number of images loaded: ', len( images ) )
    print( 'Intensity max value:', np.amax(img) )
    print( 'Intensity min value:', np.amin(img) )


def load_all_images( images_dir ):
    '''Load all images from directory

    Args:
        images_dir (str): Directory with images to load

    Returns:
        names (list[str]): List with loaded images names.
        images (list[ndarray]): List with images (multidimensional array).
        nrows (list[int]): List with number of rows of each image.
        mcols (list[int]): List with number of cols of each image.
    '''
    image_files = os.listdir( images_dir )
    # Sort files in alphabetical order
    image_files.sort()
    names = []
    rows = []
    cols = []
    images = []
    print( '-- Loading all images...' )
    for image_index, image_name in enumerate( image_files ):
        image_file = os.path.join( images_dir, image_name )
        image_array = ndimage.imread( image_file ).astype(float)
        names.append( image_name )
        rows.append( len(image_array) )
        cols.append( len(image_array[0]) )
        images.append( image_array )
    return names, images, rows, cols


def save_default( names, images, nrows, ncols, nchannels ):
    '''Load images with nrows, ncols and nchannels.
    In some datasets the images doesn't have equal number of rows and columns.

    Args:
        names (list[str]): List with all images names.
        images (list[ndarray]): List with images (multidimensional array).
        nrows (int): Number of rows of the images.
        ncols (int): Number of cols of the images.
        nchannels (int): Number of channels of the images.

    Returns:
        new_names (list[str]): List with loaded images names.
        new_images (numpy array): Numpy array with nrows,ncols images.
        images_index (numpy array): Numpy array with the selected images index from input images
    '''
    new_names = []
    new_images = []
    images_index = []
    print( '-- Selecting (', nrows, 'x', ncols, ') images...' )
    for i in range( len(images) ):
        image_shape = images[i].shape
        if len(image_shape) == 3 and image_shape[2] == nchannels:
            if nrows == len(images[i]) and ncols == len(images[i][0]) :
                images_index.append( i )
                new_names.append( names[i] )
                new_images.append( images[i] )
    new_images = np.array( new_images )
    return new_names, new_images, images_index


def choose_images( names, images, rows_list, cols_list, nchannels):
    '''Analyse the images and load those with the most frequent nrows and ncols.

    In some datasets the images doesn't have equal number of rows and columns.
    This function separate the images in subsets based in their nrows and ncols...'
    Then we pick the largest subset.

    Args:
        names (list[str]): List with all images names.
        images (list[ndarray]): List with images (multidimensional array).
        rows_list (list[int]): List with the number of rows of each image.
        cols_list (list[int]): List with the number of columns of each image.
        nchannels (int): nchannels (int): Number of channels of the images.

    Returns:
        new_names (list[str]): List with loaded images names.
        new_images (numpy array): Numpy array with nrows,ncols images.
        nrows (int): Most frequent number of rows.
        mcols (int): most frequent number of cols.
        images_index (numpy array): Numpy array with the selected images index from input images
    '''
    unique, counts = np.unique( rows_list, return_counts = True )
    print( 'Frequency of the number of rows: ' )
    print( np.asarray((unique, counts)).T )
    index = np.argmax(counts)
    nrows = unique[index]

    unique, counts = np.unique( cols_list, return_counts = True )
    print( 'Frequency of the number of cols: ' )
    print( np.asarray((unique, counts)).T )
    index = np.argmax(counts)
    ncols = unique[index]

    images_index = range( len(images) )
    # Remove images with different number of nrows x ncols
    for i in reversed( range(len(images)) ):
        image_shape = images[i].shape
        if len(image_shape) != 3 or image_shape[2] != nchannels:
            images.pop(i)
            names.pop(i)
            images_index.pop(i)
        elif( nrows != len(images[i]) or ncols != len(images[i][0]) ):
            images.pop(i)
            names.pop(i)
            images_index.pop(i)

    print( 'Number of rows more frequent:', nrows )
    print( 'Number of cols more frequent:', ncols )

    images = np.array( images )
    return names, images, nrows, ncols, images_index


def resize_images( names, images, nrows, ncols, nchannels ):
    '''Resize and load images with nrows, ncols and nchannels.

    This function will try to resize the images to (nrows,ncols).

    Args:
        names (list[str]): List with all images names.
        images (list[ndarray]): List with images (multidimensional array).
        nrows (int): Number of rows of the images.
        ncols (int): Number of cols of the images.
        nchannels (int): Number of channels of the images.

    Returns:
        new_names (list[str]): List with loaded images names
        new_images (numpy array): Numpy array with nrows,ncols images
        images_index (numpy array): Numpy array with the selected images index from input images
    '''
    new_names = []
    new_images = []
    images_index = []
    for i in range( len(images) ):
        image_shape = images[i].shape
        if len(image_shape) == 3 and image_shape[2] == nchannels:
            if image_shape[0] != nrows or image_shape[1] != ncols:
                images[i] = imresize( images[i], [nrows, ncols], interp = 'bilinear' )
            images_index.append( i )
            new_names.append( names[i] )
            new_images.append( images[i] )
    images_index = np.array( images_index )
    new_images = np.array( new_images )
    return new_names, new_images, images_index


def reduce_gt(grid, ground_truth):
    print( 'Creating grid for the ground truth...' )
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
    images = images.reshape( (-1, nrows, ncols, num_channels) ).astype(np.float32)

    # Separate dataset in training, validation and test
    nimages_training = int( 0.6 * nimages )
    nimages_validation = int( 0.2 * nimages )
    nimages_test = int( 0.2 * nimages )

    images_training = images[0:nimages_training, :]
    # images_validation
    # images_test

    gt_nrows = ground_truth.shape[1]
    gt_ncols = ground_truth.shape[2]
    output_size = gt_nrows * gt_ncols
    ground_truth = ground_truth.reshape( (-1, gt_nrows * gt_ncols) ).astype(np.float32)
    ground_truth_training = ground_truth[0:nimages_training, :]
    # ground_truth_validation
    # ground_truth_test

    images_info = nrows, ncols, num_channels, output_size
    data_size = nimages_training, nimages_validation, nimages_test
    input_data = images_training, ground_truth_training

    return images_info, data_size, input_data
