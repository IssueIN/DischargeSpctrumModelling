import os
import numpy as np
import matplotlib.pyplot as plt
from spectrum.utils import convert_files, get_img_names_from_folder, get_spectrual_line
from collections import defaultdict

folder_name = os.path.join('21_11', 'wholespectrum')
img_names_b16, img_names_tiff, output_names = get_img_names_from_folder(os.path.join('data', folder_name), suffix='.tif')

output_folder = os.path.join('output', folder_name)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    convert_files(img_names_b16, img_names_tiff, folder_name)

sum_grouped = defaultdict(list)

for i, output_name in enumerate(output_names):
    output_path = os.path.join('output', folder_name, output_name)
    
    sum, x = get_spectrual_line(output_path)
    
    for x_val, sum_val in zip(x, sum):
        sum_grouped[x_val].append(sum_val)

sum_means = []
sum_err = []
x_list = []

for x, sums in sum_grouped.items():
    x_list.append(float(x))
    sum_means.append(np.mean(sums))
    sum_err.append(np.std(sums))

sorted_indices = np.argsort(x_list)
x_list = np.array(x_list)[sorted_indices]
sum_means = np.array(sum_means)[sorted_indices]
sum_err = np.array(sum_err)[sorted_indices]

plt.errorbar(x_list, sum_means, yerr=sum_err)
plt.show()