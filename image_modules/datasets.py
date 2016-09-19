from image_modules import load_images
from image_modules import load_ground_truth
from image_modules import make_grid
# from tf_modules import udacity


# All datasets inherits from DatasetBase
class DatasetBase:
    def __init__(self):
        pass

    def load_images(self, training_dir, analyse = False ):
        if( analyse ):
            self.names, self.images, self.nrows, self.ncols = load_images.load_and_analyse( training_dir )
        else:
            self.names, self.images = load_images.load( training_dir, self.nrows, self.ncols )



class SynthiaDataset( DatasetBase ):
    def __init__(self, nrows = 720, ncols = 960):
        self.nrows = nrows
        self.ncols = ncols

    # Synthia dataset doesn't have a bb, each pixel has a class.
    # Thus, we don't need to load label files, we load the ground truth straight from the txt file.
    def load_gt(self, gt_dir, object_name):
        self.ground_truth = load_ground_truth.load_synthia_gt(gt_dir, object_name, self.nrows, self.ncols)



class KittiDataset( DatasetBase ):
    def __init__(self, nrows = 375, ncols = 1242):
        self.nrows = nrows
        self.ncols = ncols

    def load_gt(self, labels_dir, object_name ):
        self.labels = load_ground_truth.load_kitti_labels( labels_dir, object_name )
        self.bb = load_ground_truth.get_bb_pixels( self.names, self.labels, self.nrows, self.ncols )