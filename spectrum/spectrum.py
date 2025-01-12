import numpy as np
import cv2  #pip install opencv-python
import matplotlib.pyplot as plt
import os
from spectrum.utils import get_spectrual_line, read_pco_img

folder_name = 'data/Canon'
output_folder = 'output/Canon'
os.makedirs(output_folder, exist_ok=True)

img_names = [
    '1_modified.JPG'
]

img_path = os.path.join('data', '29_11', 'take4_00068.b16')

sum, x = get_spectrual_line(img_path)

# plt.scatter(x, sum, label='Spectral Line', s=2)
# # plt.plot(x, sum, label='Spectral Line')
# # plt.plot(peaks, sum[peaks], "rx", label='Peaks')
# plt.xlabel('pixel number')
# plt.ylabel('Spectrual intensity (a.u.)')

# plt.grid()
# plt.legend()
# plt.show()

img_path = 'output/21_11_plasma_first_attempt/plasmaexp150_00014_00000.tif'
img_path = os.path.join('data', '29_11', 'take4_00068.b16')
sum, x_vals = get_spectrual_line(img_path)
plt.plot(x_vals, sum)
plt.xlabel('pixel number (#)')
plt.ylabel('spectral intensity (a. u.)')
plt.grid()
plt.show()

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