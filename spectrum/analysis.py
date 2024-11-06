import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

img_path = 'data/yellow_doublet_00005.JPG'
output_path = 'output/1028_2.jpg'

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

def linear_func(x, k, b):
    return k * x + b

sum, x = get_spectrual_line(img_path)

sum = np.flip(sum)

peaks, _ = find_peaks(sum, height=300000)
peaks_modified = np.array([322, 564, 749, 1421, 1621])
wavelengths = np.array([365, 405, 436, 546, 579])

popt, pcov = curve_fit(linear_func, peaks_modified, wavelengths)
slope, intercept = popt

wl_list = linear_func(x, slope, intercept)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

ax1.plot(x,wl_list, label='Wavelength Calibration')
ax1.scatter(peaks_modified, wavelengths, label='Calibration Points')
ax1.set_xlabel('Pixel Position')
ax1.set_ylabel('Wavelength (nm)')
ax1.legend()
ax1.grid()

ax2.plot(wl_list, sum, label='Spectral Line')
ax2.plot(wavelengths, sum[peaks_modified], "bo", label='Peaks')
# plt.plot(x, sum, label='Spectral Line')
# plt.plot(peaks, sum[peaks], "rx", label='Peaks')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Spectrual intensity (a.u.)')

plt.tight_layout()
plt.show()

fig.savefig('output/spectrum_calibration.jpg')