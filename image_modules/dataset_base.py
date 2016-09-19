from image_modules import load_images
# from image_modules import make_grid
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