from image_modules import load_images
import numpy as np


# All datasets inherits from DatasetBase
class DatasetBase:
    def __init__(self):
        pass

    def load_images( self, training_dir, analyse = False ):
        if( analyse ):
            self.names, self.images, self.nrows, self.ncols = load_images.load_and_analyse( training_dir )
        else:
            self.names, self.images = load_images.load( training_dir, self.nrows, self.ncols )

    # divide the image in grid blocs
    def create_gt_grid( self, grid = [25,27] ):
        print( '-- Creating grid for the ground truth...' )
        n_bb = self.ground_truth.shape[0]
        horizontal_size = self.ground_truth.shape[1] / grid[0]
        vertical_size = self.ground_truth.shape[2] / grid[1]
        self.reduced_ground_truth = np.ndarray( (n_bb , horizontal_size, vertical_size), float)
        self.reduced_ground_truth.fill(0)
        for k in range( n_bb ):
            for i in range ( horizontal_size ):
                row = i * grid[0]
                for j in range ( vertical_size ):
                    column = j * grid[1]
                    box = self.ground_truth[k, row:(row + grid[0]), column:(column + grid[1]) ]
                    n_ones = np.sum( box.reshape(1, box.size) == 1 )
                    percentage = np.divide( n_ones, grid[0] * grid[1] ,dtype = float )
                    # if more the 90% f the box have ones => the area will keep the value
                    if ( percentage >= 0.9 ):
                        self.reduced_ground_truth[k, i, j] = 1
        return 0

    def format_dataset( self ):
        # Transform each 2D image in an unidimentional array
        nimages = len( self.images )
        num_channels = 1
        self.images = self.images.reshape( (-1, self.nrows, self.ncols, num_channels)).astype(np.float32)

        gt_nrows = self.reduced_ground_truth.shape[1]
        gt_ncols = self.reduced_ground_truth.shape[2]
        output_size = gt_nrows * gt_ncols
        self.reduced_ground_truth = self.reduced_ground_truth.reshape( (-1, gt_nrows * gt_ncols)).astype(np.float32)

        # Separate dataset in training, validation and test
        # training = 60%
        # validation and test = 20%
        nimages_training = int( 0.6 * nimages )
        nimages_validation = int( 0.2 * nimages )
        nimages_test = int( 0.2 * nimages )

        images_training = self.images[0:nimages_training, :]
        # images_validation
        # images_test

        ground_truth_training = self.reduced_ground_truth[0:nimages_training, :]
        # bb_validation
        # bb_test

        images_info = self.nrows, self.ncols, num_channels, output_size
        data_size = nimages_training, nimages_validation, nimages_test
        input_data = images_training, ground_truth_training

        return images_info, data_size, input_data
