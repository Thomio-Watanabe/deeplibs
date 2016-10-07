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



def get_img_directory():
    imgs_dir_name.set( filedialog.askdirectory() )

def get_gt_directory():
    gt_dir_name.set( filedialog.askdirectory() )

def close_window(): 
    window.destroy()

def train_model():
    if dataset_name.get() == 'mnist':
        dataset = mnist_dataset.MnistDataset()
    if dataset_name.get() == 'cifar10':
        dataset = cifar10_dataset.Cifar10Dataset()

    dataset.load_images( imgs_dir_name.get() )
    dataset.load_labels( gt_dir_name.get() )
    dataset.format_dataset()

    if model_name.get() == 'LeNet':
        LeNet.model( dataset )
    if model_name.get() == 'AlexNet':
        AlexNet.model( dataset)

    print('-- Finished training model.')



window = tk.Tk()
window.wm_title('DeepLibs')
window.eval('tk::PlaceWindow %s center' % window.winfo_pathname(window.winfo_id()))

dataset_label = tk.Label( window, width = 12, text="Datasets:", fg="black")
dataset_label.grid(row=0, column=0)
dataset_name = tk.StringVar( )
dataset_name.set('Select dataset')
dataset_cbox = ttk.Combobox( window, width = 15, textvariable=dataset_name, state='readonly' )
dataset_cbox['values'] = ('mnist', 'cifar10')
dataset_cbox.grid( row=0, column=1, padx=2, pady=2 )

model_label = tk.Label( window, width = 12, text="Models:", fg="black")
model_label.grid( row=1, column=0 )
model_name = tk.StringVar( )
model_name.set('Select model')
model_cbox = ttk.Combobox( window, width = 15, textvariable=model_name, state='readonly' )
model_cbox['values'] = ('LeNet', 'AlexNet')
model_cbox.grid( row=1, column=1, padx=2, pady=2 )

imgs_directory = tk.Button(window, text ="Images Dir", width = 12, command = get_img_directory).grid( row=3, column = 0 )
imgs_dir_name = tk.StringVar( )
imgs_dir_name.set('Image dir')
imgs_dir_entry = tk.Entry( window, width = 16, textvariable=imgs_dir_name, state='readonly' )
imgs_dir_entry.grid( row=3, column=1, padx=2, pady=2 )

gt_directory = tk.Button(window, text ="Ground Truth Dir", width = 12, command = get_gt_directory).grid( row=4, column = 0 )
gt_dir_name = tk.StringVar( )
gt_dir_name.set('Ground truth dir')
gt_dir_entry = tk.Entry( window, width = 16, textvariable=gt_dir_name, state='readonly' )
gt_dir_entry.grid( row=4, column=1, padx=2, pady=2 )

close = tk.Button(window, text ="Close", width = 12, command = close_window ).grid( row=5, column = 0 )
run_training = tk.Button(window, text ="Train model", width = 12, command = train_model).grid( row=5, column = 1 )

window.mainloop()

