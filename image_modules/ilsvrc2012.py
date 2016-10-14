from image_modules import images_dataset
import numpy as np
import os


class ILSVRC2012Dataset( images_dataset.ImagesDataset ):
    def __init__( self, nrows = 375, ncols = 500 ):
        print('-------------------------')
        print('-- ILSVRC 2012 Dataset --')
        print('-------------------------')
        self.num_rows = nrows
        self.num_cols = ncols
        self.num_labels = 1000
        self.num_channels = 3

    def load_labels( self, gt_dir ):
        self.labels = load_labels( gt_dir )

    def format( self,
                gray_scale = False,
                normalize = True,
                batch_size = 16,
                num_epochs = 10,
                eval_batch_size = 16,
                eval_frequency = 100):
        if gray_scale:
            self.images = images_dataset.rgb2gray( self.images )
            self.num_channels = 1

        if normalize:
            self.images = images_dataset.normalize_images( self.images )

        # Format labels as 1D array of 1 element
        self.labels = self.labels.ravel()
        # Get labels correspondent to each image
        self.labels = self.labels[ self.images_index ]

        # Format images as 1D array of 4 elements
        _num_images = len( self.images_index )
        self.images = self.images.reshape( _num_images, self.num_rows, self.num_cols, self.num_channels ).astype(np.float32)

        # 20% os the images and labels go for validation and test
        validation_size = int( _num_images / 10 )
        test_size = int( _num_images / 10 )

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

        images_dataset.images_info( self.images )


def load_labels( gt_dir ):
    gt_name = 'ILSVRC2012_validation_ground_truth.txt'
    gt_file = os.path.join( gt_dir, gt_name)
    with open( gt_file ) as fileHandle:
        array = [[int(x) for x in line.split()] for line in fileHandle]
    ground_truth = np.array( array )
    return ground_truth

