import numpy as np
import cv2  #pip install opencv-python
import matplotlib.pyplot as plt
import os

folder_name = 'data/Canon'
output_folder = 'output/Canon'
os.makedirs(output_folder, exist_ok=True)

img_names = [
    '1_modified.JPG'
]

img_path = 'data/Canon/1_modified.JPG'

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

    return sum, x_vals

sum, x = get_spectrual_line(img_path)

# plt.scatter(x, sum, label='Spectral Line', s=2)
# # plt.plot(x, sum, label='Spectral Line')
# # plt.plot(peaks, sum[peaks], "rx", label='Peaks')
# plt.xlabel('pixel number')
# plt.ylabel('Spectrual intensity (a.u.)')

# plt.grid()
# plt.legend()
# plt.show()

# img_path = 'data/11_11_FWHM_cali/0.5rota_00005.JPG'
# sum, x_vals = get_spectrual_line(img_path)
# plt.plot(x_vals, sum)
# plt.xlabel('pixel number (#)')
# plt.ylabel('spectral intensity (a. u.)')
# plt.grid()
# plt.show()

# for img_name in img_names:
#     img_path = os.path.join(folder_name, img_name)
#     sum_col, x_vals = get_spectrual_line(img_path)
    
#     plt.figure()
#     plt.plot(x_vals, sum_col)
#     plt.xlabel('Pixel Number (#)')
#     plt.ylabel('Spectral Intensity (a.u.)')
#     plt.title(f'Spectral Line Plot: {img_name}')
#     plt.grid()
    
#     output_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}.png")
#     plt.savefig(output_path)
#     plt.close()  # Close the plot to prevent overlap in the next iteration

#     print(f"Plot saved for {img_name} at {output_path}")