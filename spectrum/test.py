import os
import numpy as np
import matplotlib.pyplot as plt
from spectrum.utils import convert_files, get_img_names_from_folder, get_spectrual_line

folder_name = os.path.join('22_11', 'imagingCalibration')
img_names_b16, img_names_tiff, output_names = get_img_names_from_folder(os.path.join('data', folder_name), suffix='.tif')

output_folder = os.path.join('output', folder_name)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    convert_files(img_names_b16, img_names_tiff, folder_name)

