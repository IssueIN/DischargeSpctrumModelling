import numpy as np
import cv2
import matplotlib.pyplot as plt

img_path = 'data/whole_spectrum_00005.JPG'

def sum_up_col(grid):
    num_cols = grid.shape[1]
    sum_cols = np.zeros(num_cols)

    for i in range(num_cols):
        sum_cols[i] = grid[:,i].sum()
    
    return sum_cols

def get_spectrual_line(img_path):
    img_grid = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    sum = sum_up_col(img_grid)
    x_vals = np.arange(len(sum))

    # fig = plt.figure()
    # plt.scatter(x_vals, sum, s=2)
    # plt.xlabel('position (a. u.)')
    # plt.ylabel('pixel value (a.u.)')
    # plt.grid()
    # # fig.savefig(output_path)

    return sum, x_vals

sum, x_vals = get_spectrual_line(img_path)
plt.scatter(x_vals, sum, s=2)
plt.xlabel('pixel number (#)')
plt.ylabel('spectral intensity (a. u.)')
plt.grid()
plt.show()