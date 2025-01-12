import numpy as np
import cv2
from pco_tools import pco_reader as pco
from utils.fileConversion import file_conversion
import matplotlib.pyplot as plt
import os
from scipy.ndimage import rotate

def read_pco_img(img_path):
    img = pco.load(img_path)
    return img

def sum_up_col(grid):
    num_cols = grid.shape[1]
    sum_cols = np.zeros(num_cols)

    for i in range(num_cols):
        sum_cols[i] = grid[:,i].sum()
    
    return sum_cols

def get_spectrual_line(img_path):
    #img_grid = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_grid = read_pco_img(img_path)
    
    sum = sum_up_col(img_grid)
    x_vals = np.arange(len(sum))

    return sum, x_vals

def get_img_names_from_folder(folder):
    img_names_b16 = [file for file in os.listdir(folder) if file.endswith('b16')]
    return img_names_b16

def extract_prefix(filename):
    return filename.split('_')[0]

def convert_files(img_names_b16, img_names_jpg, folder_name):
    for i, img_name_b16 in enumerate(img_names_b16):
        img_path_b16 = os.path.join(folder_name, img_name_b16)
        img_path_jpg = os.path.join(folder_name, img_names_jpg[i])

        file_conversion(
        input_filename = img_path_b16, 
        output_filename = img_path_jpg)

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def gaussian(x, a, x0, sigma, c):
    return a * np.exp(-((x-x0)**2) / (2 * sigma**2)) + c

def rotate_image_grid(array, angle):
    rotated_array = rotate(array, angle, reshape=False, mode = 'nearest')
    return rotated_array

def crop_image(array, top_left, bottom_right):
    row_start, col_start = top_left
    row_end, col_end = bottom_right
    return array[row_start:row_end, col_start:col_end]