import numpy as np

# divide the image in grid = [25 x 27]
def generate( bb, grid = [25,27] ):
    n_bb = bb.shape[0]
    horizontal_size = bb.shape[1] / grid[0]
    vertical_size = bb.shape[2] / grid[1]
    reduced_bb = np.ndarray( (n_bb , horizontal_size, vertical_size), float)
    reduced_bb.fill(0)
    for k in range( n_bb ):
        for i in range ( horizontal_size ):
            row = i * grid[0]
            for j in range ( vertical_size ):
                column = j * grid[1]
                box = bb[k, row:(row + grid[0]), column:(column + grid[1]) ]
                n_ones = np.sum( box.reshape(1, box.size) == 1 )
                percentage = np.divide( n_ones, grid[0] * grid[1] ,dtype = float )
                # if more the 90% f the box have ones => the area will keep the value
                if ( percentage >= 0.9 ):
                    reduced_bb[k, i, j] = 1
    return reduced_bb;