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
        print '-- Creating grid for the ground truth...'
        n_bb = self.ground_truth.shape[0]
        horizontal_size = self.ground_truth.shape[1] / grid[0]
        vertical_size = self.ground_truth.shape[2] / grid[1]
        reduced_bb = np.ndarray( (n_bb , horizontal_size, vertical_size), float)
        reduced_bb.fill(0)
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
                        reduced_bb[k, i, j] = 1
        return 0
