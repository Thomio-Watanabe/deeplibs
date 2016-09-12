import numpy as np


# Return bouding box pixels
def get_bb_pixels( names, labels, nrows, ncols):
    bb = []
    for i in range( len(names) ):
        zeros = np.zeros( [nrows,ncols] )
        for j in range( len(labels) ):
            if( names[i].strip('.png')  == labels[j][0].strip('.txt') ):
                x = labels[j][1]
                x = [ int( float(k) ) for k in x ]
                zeros[ x[0]:x[1], x[2]:x[3] ] = 1
        bb.append(zeros)
    return bb;