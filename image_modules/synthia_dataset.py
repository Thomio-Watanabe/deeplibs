from image_modules import dataset_base
import numpy as np
import os


class SynthiaDataset( dataset_base.DatasetBase ):
    def __init__(self, nrows = 720, ncols = 960):
        self.nrows = nrows
        self.ncols = ncols

    # Synthia dataset doesn't have a bb, each pixel has a class.
    # Thus, we don't need to load label files, we load the ground truth straight from the txt file.
    def load_gt(self, gt_dir, object_name):
        self.ground_truth = load(  gt_dir, object_name )


def load( gt_dir, object_name ):
    # Dcitionary with synthia object classes
    object_class = {'void': 0,
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
    if object_name not in object_class:
        print( '-- Object name ', object_name,' not found in Synthia dataset.' )
        print( '-- Possible options are: ', object_class.keys() )
        raise SystemExit

    object_ID = object_class[object_name]
    print( '-- Loading ' + object_name + ' ground truth... ' )

    gt_files = os.listdir( gt_dir )
    ground_truth_list = []
    for gt_index, gt_name in enumerate( gt_files ):
        gt_file = os.path.join( gt_dir, gt_name)
        with open( gt_file ) as fileHandle:
            array = [[int(x) for x in line.split()] for line in fileHandle]
            ground_truth_list.append( array )

            ground_truth = np.array( ground_truth_list )
            # Pixels that don't belong to the object receive 0
            for i in range( len(ground_truth_list) ):
                non_object_index = np.where( ground_truth[i] != object_ID )
                ground_truth[i][non_object_index] = 0
    return ground_truth

