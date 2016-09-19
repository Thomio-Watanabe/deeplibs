from scipy import ndimage
import numpy as np
import os


def rgb2grey(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# Normalize the matrix
def norm_image( array_2D ):
    pixel_depth = 255.0
    return ( array_2D - (pixel_depth) / 2 ) / pixel_depth


# Kitti ( nrows = 375 x ncols = 1242 )
# Synthia ( nrows = 720 x ncols = 960 )
def analyse_imgs( images ):
    img = images[0]
    print '-- Image array format:', img.shape
    print '-- Total number of images loaded: ', len( images )
    # print '-- Images dimensions:', img.ndim    # 3 dimensions RGB -> MxNx3
    # print '-- Number of rows:', len(img)
    # print '-- Number of cols:', len(img[0])
    print '-- Intensity max value:', np.amax(img)
    print '-- Intensity min value:', np.amin(img)


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
    print '-- Loading (', nrows, 'x', ncols, ') images...'
    for image_index, image_name in enumerate( image_files ):
        image_file = os.path.join( training_dir, image_name )
        image_array = ndimage.imread( image_file ).astype(float)
        if( (nrows == len(image_array)) and (ncols == len(image_array[0])) ):
            if( grey_scale ):
                image_array = rgb2grey( image_array )
            norm_image_array = norm_image( image_array )
            names.append( image_name )
            images.append( norm_image_array )
    analyse_imgs( images )
    return names, images


def load_and_analyse( training_dir, grey_scale = True ):
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
    print '-- Loading all images...'
    for image_index, image_name in enumerate( image_files ):
        names.append( image_name )
        image_file = os.path.join( training_dir, image_name )
        image_array = ndimage.imread( image_file ).astype(float)
        rows.append( len(image_array) )
        cols.append( len(image_array[0]) )
        images.append( image_array )

    unique, counts = np.unique( rows, return_counts = True )
    index = np.argmax(counts)
    print '-- Frequency of the number of rows: '
    print np.asarray((unique, counts)).T
    nrows = unique[index]

    unique, counts = np.unique( cols, return_counts = True )
    index = np.argmax(counts)
    print '-- Frequency of the number of cols: '
    print np.asarray((unique, counts)).T
    ncols = unique[index]

    # Remove images with different number of nrows x ncols
    for i in reversed( range(len(images)) ):
        if( nrows != len(images[i]) or ncols != len(images[i][0]) ):
            images.pop(i)
            names.pop(i)

    print '-- Number of rows more frequent:', nrows
    print '-- Number of cols more frequent:', ncols

    for i in range( len(images) ):
        if( grey_scale ):
            images[i] = rgb2grey( images[i] )
        images[i] = norm_image( images[i] )

    analyse_imgs( images )

    return names, images, nrows, ncols
