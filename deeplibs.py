#!/usr/bin/python
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk


def dataset_callback(*dataset_name):
    print( "New dataset name!" )

def model_callback(*model_name):
    print( "New model name!" )

def get_img_directory():
    imgs_dir_name.set( filedialog.askdirectory() )

def get_gt_directory():
    gt_dir_name.set( filedialog.askdirectory() )

def close_window(): 
    window.destroy()


window = tk.Tk()
window.wm_title('DeepLibs')
window.eval('tk::PlaceWindow %s center' % window.winfo_pathname(window.winfo_id()))

dataset_label = tk.Label( window, width = 12, text="Datasets:", fg="black")
dataset_label.grid(row=0, column=0)
dataset_name = tk.StringVar( )
dataset_name.set('Select dataset')
dataset_name.trace("w", dataset_callback)
dataset_cbox = ttk.Combobox( window, width = 15, textvariable=dataset_name, state='readonly' )
dataset_cbox['values'] = ('mnist', 'cifar10')
dataset_cbox.grid( row=0, column=1, padx=2, pady=2 )

model_label = tk.Label( window, width = 12, text="Models:", fg="black")
model_label.grid( row=1, column=0 )
model_name = tk.StringVar( )
model_name.set('Select model')
model_name.trace("w", model_callback)
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

close_window = tk.Button(window, text ="Close", width = 12, command = close_window ).grid( row=5, column = 0 )
run_training = tk.Button(window, text ="Train model", width = 12).grid( row=5, column = 1 )

window.mainloop()

