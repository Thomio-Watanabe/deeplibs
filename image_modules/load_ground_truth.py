import numpy as np
import os


def load_kitti_labels( labels_dir, object_name ):
    # Positions of the bouding box pixels inside each label file
    col_first = 4
    row_first = 5
    col_last = 6
    row_last = 7
    label_files = os.listdir( labels_dir )
    label_dataset = []
    print 'Loading ' + object_name + ' ground truth... '
    for label_index, label_name in enumerate( label_files ):
        label_file = os.path.join( labels_dir, label_name)
        # print 'Loading label file:', label_file # -> todo save this info to file
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



def load_synthia_gt(gt_dir, object_name, nrows, ncols):
    # Dcitionary with synthia object classes
    object_class =  {'void': 0,
                     'Sky': 1,
                     'Building': 2,
                     'Road': 3,
                     'Sidewalk': 4,
                     'Fence': 5,
                     'Vegetation': 6,
                     'Pole': 7,
                     'Car': 8,
                     'Sign': 9,
                     'Pedestrian': 10,
                     'Cyclist': 11}

    # print error if key doesn't exit:
    if( not object_class.has_key(object_name) ):
        print '-- Object name ', object_name,' not found in Synthia dataset.'
        print '-- Possible options are: ', object_class.keys()
        raise SystemExit

    object_ID = object_class[object_name] #object_class.setdefault(object_name, -1)
    print '-- Loading ' + object_name + ' ground truth... '

    gt_files = os.listdir( gt_dir )
    ground_truth_list = []
    for gt_index, gt_name in enumerate( gt_files ):
        gt_file = os.path.join( gt_dir, gt_name)
        with open( gt_file ) as fileHandle:
            array = [[int(x) for x in line.split()] for line in fileHandle]
            ground_truth_list.append( array )

    ground_truth_array = np.array( ground_truth_list )
    # Pixels that don't belong to the object receive 0
    for i in range( len(ground_truth_list) ):
        non_object_index = np.where( ground_truth_array[i] != object_ID )
        ground_truth_array[i][non_object_index] = 0

    return ground_truth_list

