import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from pco_tools import pco_reader as pco
import os
from scipy.ndimage import rotate
import json

folder_path = os.path.join('data', 'Spatially_resolved_Mercury_line_calibration')
img_path = os.path.join(folder_path, '1_00005.b16')
output_path = 'output/canon1.JPG'

def read_pco_img(img_path):
    img = pco.load(img_path)
    return img

def rotate_image_grid(array, angle):
    rotated_array = rotate(array, angle, reshape=False, mode = 'nearest')
    return rotated_array

def crop_image(array, top_left, bottom_right):
    row_start, col_start = top_left
    row_end, col_end = bottom_right
    return array[row_start:row_end, col_start:col_end]

def sum_up_col(grid):
    num_cols = grid.shape[1]
    sum_cols = np.zeros(num_cols)

    for i in range(num_cols):
        sum_cols[i] = grid[:,i].sum()
    
    return sum_cols

def display_array_as_image(array, cmap="gray"):
    """
    Displays a 2D array as an image.
    
    Parameters:
    - array: 2D numpy array to display.
    - cmap: Colormap for visualizing the array as an image (default is "gray").
    """
    plt.figure(figsize=(7, 4))
    plt.imshow(array, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Intensity", orientation='horizontal')
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.tight_layout()
    plt.show()


def get_spectrual_line(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"The file '{img_path}' does not exist.")
    
    # Load the image
    img_grid = read_pco_img(img_path)
    if img_grid is None:
        raise ValueError(f"The file '{img_path}' could not be read as an image.")

    sum = sum_up_col(img_grid)
    x_vals = np.arange(len(sum))

    # fig = plt.figure()
    # plt.scatter(x_vals, sum, s=2)
    # plt.xlabel('position (a. u.)')
    # plt.ylabel('pixel value (a.u.)')
    # plt.grid()
    # # fig.savefig(output_path)

    return sum, x_vals

def linear_func(x, k, b):
    return k * x + b

# sum, x = get_spectrual_line(img_path)

angle = -2.2
top = 800
left = 200
bottom = 1300
right = 20000

# with open(os.path.join(folder_path, 'config.json'), "r") as f:
#         config = json.load(f)  
# wavelengths = np.array(config["wavelengths"])
# rotation_angle = config["rotation_angle"]
# top_left = tuple(config["top_left"])
# bottom_right = tuple(config["bottom_right"])

img = read_pco_img(img_path)
# img = rotate_image_grid(img, rotation_angle)
img = rotate_image_grid(img, angle)
top_left = (top, left)
bottom_right = (bottom, right)
img = crop_image(img, top_left, bottom_right)
display_array_as_image(img)
sum = sum_up_col(img)
sum = np.flip(sum)

x_vals = np.arange(len(sum))

plt.plot(x_vals, sum)

peaks, _ = find_peaks(sum, threshold=300000)
print(peaks)
# peaks_modified = np.array([449, 639, 1310, 1497, 1509])
# peaks_modified = peaks
peaks_modified = np.array([1185, 1480])
wavelengths = np.array([546, 579])#365, 405, 436, 546, 576, 579

popt, pcov = curve_fit(linear_func, peaks_modified, wavelengths)
slope, intercept = popt

wl_list = linear_func(x_vals, slope, intercept)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

ax1.plot(x_vals,wl_list, label='Wavelength Calibration', c='black')
ax1.scatter(peaks_modified, wavelengths, label='Calibration Points', c='black')
ax1.set_xlabel('Pixel Position')
ax1.set_ylabel('Wavelength (nm)')
ax1.legend()
ax1.grid()

ax2.plot(wl_list, sum, label='Spectral Line', c='black')
ax2.plot(wavelengths, sum[peaks_modified], "bo", label='Peaks', c='blue')
# ax2.plot(x, sum, label='Spectral Line')
# ax2.plot(peaks_modified, sum[peaks_modified], "rx", label='Peaks')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Spectrual intensity (a.u.)')
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()

# fig.savefig(os.path.join('fig', 'imaging_mercury_line_calibration(spatialling_resolved)'), dpi=300)

wl_list = wl_list.tolist()
metadata = {
    "rotation_angle": angle,
    "top_left": [top, left],         # Tuples are stored as lists in JSON
    "bottom_right": [bottom, right],
    "wavelengths": wl_list  # Example array of wavelengths
}

with open(os.path.join(folder_path, 'config.json'), "w") as f:
    json.dump(metadata, f, indent=4)