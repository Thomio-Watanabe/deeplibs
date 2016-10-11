from image_modules.images_dataset import ImagesDataset
import numpy as np
import os


class ILSVRC2012Dataset( ImagesDataset ):
    def __init__( self, nrows = 375, ncols = 500 ):
        print('-------------------------')
        print('-- ILSVRC 2012 Dataset --')
        print('-------------------------')
        self.model_type = 'classification'
        self.num_rows = nrows
        self.num_cols = ncols
        self.num_labels = 1000
        self.num_channels = 3

    def load_labels( self, gt_dir ):
        self.labels = load_labels( gt_dir )

    def format( self,
                batch_size = 64,
                num_epochs = 10,
                eval_batch_size = 64,
                eval_frequency = 100):
        num_images = len( self.images )
        self.labels = self.labels[:num_images]
        self.labels = self.labels.reshape( num_images )

        # 20% os the images and labels go for validation and test
        validation_size = int( num_images / 10 )
        test_size = int( num_images / 10 )

        self.validation_data = self.images[:validation_size, ...]
        self.validation_labels = self.labels[:validation_size]
        self.train_data = self.images[validation_size:, ...]
        self.train_labels = self.labels[validation_size:]

        self.test_data = self.train_data[:test_size, ...]
        self.test_labels = self.train_labels[:test_size]
        self.train_data = self.train_data[test_size:, ...]
        self.train_labels = self.train_labels[test_size:]

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.eval_batch_size = eval_batch_size
        self.eval_frequency = eval_frequency


def load_labels( gt_dir ):
    gt_name = 'ILSVRC2012_validation_ground_truth.txt'
    gt_file = os.path.join( gt_dir, gt_name)
    with open( gt_file ) as fileHandle:
        array = [[int(x) for x in line.split()] for line in fileHandle]
    ground_truth = np.array( array )
    return ground_truth

