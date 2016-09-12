from scipy import ndimage
import numpy as np
import os


def rgb2grey(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# Normalize the matrix
def norm_image( array_2D ):
    pixel_depth = 255.0
    return ( array_2D - (pixel_depth) / 2 ) / pixel_depth


# Load all images
'''
REWRITE this function to:
 - add zeros if the image is small than nrows x ncols
 - remove pixels if the image is larger than nrows x ncols
'''
def load( training_dir, nrows, ncols ):
    image_files = os.listdir( training_dir )
    # index array has all the images with the parsed nrows and ncols
    names = []
    images = []
    print 'Loading (', nrows, 'x', ncols, ') images...'
    for image_index, image_name in enumerate( image_files ):
        image_file = os.path.join( training_dir, image_name )
        image_array = ndimage.imread( image_file ).astype(float)
        grey_image = rgb2grey( image_array )
        norm_grey_image = norm_image( grey_image )
        if( (nrows == len(image_array)) and (ncols == len(image_array[0])) ):
            names.append( image_name )
            images.append( norm_grey_image )
    return names, images