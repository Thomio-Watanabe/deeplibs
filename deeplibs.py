#!/usr/bin/python
try: # python2
    import Tkinter as tk
    import tkFileDialog as filedialog
    import ttk
except ImportError: # python3
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import ttk

from image_modules import mnist_dataset
from image_modules import cifar10_dataset
from tf_modules import LeNet
from tf_modules import AlexNet


class DeepLibsGUI:
    def __init__( self, window ):
        self.datasets = 'mnist', 'cifar'
        self.models = 'LeNet', 'AlexNet'

        dataset_label = tk.Label( window, width = 12, text="Datasets:", fg="black")
        dataset_label.grid(row=0, column=0)
        self.dataset_name = tk.StringVar( )
        self.dataset_name.set('Select dataset')
        dataset_cbox = ttk.Combobox( window, width = 25, textvariable=self.dataset_name, state='readonly' )
        dataset_cbox['values'] = ( self.datasets )
        dataset_cbox.grid( row=0, column=1, columnspan=2, padx=2, pady=2, sticky = tk.E )

        model_label = tk.Label( window, width = 12, text="Models:", fg="black")
        model_label.grid( row=1, column=0 )
        self.model_name = tk.StringVar( )
        self.model_name.set('Select model')
        model_cbox = ttk.Combobox( window, width = 25, textvariable=self.model_name, state='readonly' )
        model_cbox['values'] = ( self.models )
        model_cbox.grid( row=1, column=1, columnspan=2, padx=2, pady=2, sticky = tk.E )

        imgs_directory = tk.Button(window, text ="Images dir:", width = 12, command = self.get_img_directory).grid( row=2, column = 0, padx=2, pady=2 )
        self.imgs_dir_name = tk.StringVar( )
        self.imgs_dir_name.set('Image dir')
        imgs_dir_entry = tk.Entry( window, width = 30, textvariable=self.imgs_dir_name, state='readonly' )
        imgs_dir_entry.grid( row=2, column=1, columnspan=2, padx=2, pady=2, sticky = tk.E )

        gt_directory = tk.Button(window, text ="Ground truth dir:", width = 12, command = self.get_gt_directory).grid( row=3, column = 0, padx=2, pady=2 )
        self.gt_dir_name = tk.StringVar( )
        self.gt_dir_name.set('Ground truth dir')
        gt_dir_entry = tk.Entry( window, width = 30, textvariable=self.gt_dir_name, state='readonly' )
        gt_dir_entry.grid( row=3, column=1, columnspan=2, padx=2, pady=2, sticky = tk.E )

        close = tk.Button(window, text ="Close", width = 12, command = self.close_window ).grid( row=4, column = 1, padx=2, pady=2  )
        run_training = tk.Button(window, text ="Train model", width = 12, command = self.train_model).grid( row=4, column = 2, padx=2, pady=2  )

    def get_img_directory( self ):
        self.imgs_dir_name.set( filedialog.askdirectory() )

    def get_gt_directory( self ):
        self.gt_dir_name.set( filedialog.askdirectory() )

    def close_window( self ):
        window.destroy()

    def train_model( self ):
        if self.dataset_name.get() == 'mnist':
            dataset = mnist_dataset.MnistDataset()
            dataset.load_images( self.imgs_dir_name.get() )
            dataset.load_labels( self.gt_dir_name.get() )
            dataset.format_dataset()

        if self.dataset_name.get() == 'cifar10':
            dataset = cifar10_dataset.Cifar10Dataset()
            dataset.load_images( self.imgs_dir_name.get() )
            dataset.load_labels( self.gt_dir_name.get() )
            dataset.format_dataset()

        if self.model_name.get() == 'LeNet':
            LeNet.model( dataset )
        if self.model_name.get() == 'AlexNet':
            AlexNet.model( dataset )

        print('-- Finished training model.')


if __name__ == '__main__':
    window = tk.Tk()
    window.wm_title('DeepLibs')
    window.resizable(width=False, height=False)
    window.eval('tk::PlaceWindow %s center' % window.winfo_pathname(window.winfo_id()))
    DeepLibsGUI( window )
    window.mainloop()

