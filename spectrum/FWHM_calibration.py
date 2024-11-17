import numpy as np
from spectrum.spectrum import get_spectrual_line
from utils.fileConversion import file_conversion
import matplotlib.pyplot as plt
import os

def linear_approx(x, y, i, half_max):
    return x[i] + (half_max - y[i]) * ((x[i + 1] - x[i]) / (y[i + 1] - y[i]))

def FWHM(x, y):
    half_max = max(y) / 2.0
    det  = np.sign( y - half_max)
    zero_crossings = (det[0:-2] != det[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    x1 = linear_approx(x, y, zero_crossings_i[1], half_max)
    x2 = linear_approx(x, y, zero_crossings_i[0], half_max)
    FWHM = np.abs(x1 - x2)
    return x1, x2, FWHM

img_names_b16 = []
img_names_jpg = []
output_names = []
values = []

for i in range(0, 15):  
    value = i * 0.5
    values.append(value)

    formatted_value = int(value) if value.is_integer() else f"{value:.1f}"
    
    img_name_b16 = f"{formatted_value}_00002.b16"
    img_names_b16.append(img_name_b16)
    
    img_name_jpg = f"{formatted_value}.jpg"
    img_names_jpg.append(img_name_jpg)

    output_name = f"{formatted_value}_00000.jpg"
    output_names.append(output_name)

FWHM_list = []

for i, img_name_b16 in enumerate(img_names_b16):
    img_path_b16 = os.path.join('green', img_name_b16)
    img_path_jpg = os.path.join('11_12_green', img_names_jpg[i])
    output_path = os.path.join('output', '11_12_green', output_names[i])
    file_conversion(
       input_filename = img_path_b16, 
       output_filename = img_path_jpg)
    
    sum, x = get_spectrual_line(output_path)
    x1, x2, FWHM_ = FWHM(x, sum)
    FWHM_list.append(FWHM_)

plt.scatter(values, FWHM_list)
plt.xlabel('ratation (#)')
plt.ylabel('FWHM (pixel number)')
plt.grid()
plt.show()