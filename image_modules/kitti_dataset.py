from image_modules import dataset_base
import numpy as np
import os


class KittiDataset( dataset_base.DatasetBase ):
    def __init__(self, nrows = 375, ncols = 1242):
        self.nrows = nrows
        self.ncols = ncols

    def load_gt(self, labels_dir, object_name ):
        self.labels = load_kitti_labels( labels_dir, object_name )
        self.ground_truth = get_bb_pixels( self.names, self.labels, self.nrows, self.ncols )


def load_kitti_labels( labels_dir, object_name ):
    # Positions of the bouding box pixels inside each label file
    col_first = 4
    row_first = 5
    col_last = 6
    row_last = 7
    label_files = os.listdir( labels_dir )
    label_dataset = []
    print '-- Loading ' + object_name + ' ground truth... '
    for label_index, label_name in enumerate( label_files ):
        label_file = os.path.join( labels_dir, label_name)
        with open( label_file ) as fileHandle:
            file_content = fileHandle.readlines()
            # file_content is a list of strings and may have several lines
            for i in range(len(file_content)):
                obstacle = file_content[i].split()
                if ( obstacle[0] == object_name ):
                    # Create a tuple with file name and bounding box pixels
                    bounding_box = [ label_name, [obstacle[row_first], obstacle[row_last], obstacle[col_first], obstacle[col_last]] ]
                    # Save tuple to array of tuples
                    label_dataset.append( bounding_box )
    return label_dataset

# Return bouding box pixels
def get_bb_pixels( names, labels, nrows, ncols ):
    bb = []
    for i in range( len(names) ):
        zeros = np.zeros( [nrows,ncols] )
        for j in range( len(labels) ):
            if( names[i].strip('.png')  == labels[j][0].strip('.txt') ):
                x = labels[j][1]
                x = [ int( float(k) ) for k in x ]
                zeros[ x[0]:x[1], x[2]:x[3] ] = 1
        bb.append(zeros)
    bb = np.array(bb)
    return bb;

    
