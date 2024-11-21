import numpy as np
from spectrum.spectrum import get_spectrual_line
from utils.fileConversion import file_conversion
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from scipy.optimize import curve_fit

folder_name = '11_20_FWHM_Cali'

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

def generate_img_names(number, range_value):
    img_names_b16 = []
    img_names_jpg = []
    output_names = []
    values = []

    for i in range(0, range_value):  
        value = i * 0.5
        values.append(value)

        formatted_value = int(value) if value.is_integer() else f"{value:.1f}"
        
        img_name_b16 = f"{formatted_value}_0000{number}.b16"
        img_names_b16.append(img_name_b16)
        
        img_name_jpg = f"{formatted_value}.jpg"
        img_names_jpg.append(img_name_jpg)

        output_name = f"{formatted_value}_00000.jpg"
        output_names.append(output_name)

        return img_names_b16, img_names_jpg, output_names
    
def get_img_names_from_folder(folder):
    img_names_b16 = [file for file in os.listdir(folder) if file.endswith('b16')]
    img_names_jpg = [os.path.splitext(file)[0] + ".jpg" for file in img_names_b16]
    output_names = [os.path.splitext(file)[0] + "_00000.jpg" for file in img_names_b16]
    return img_names_b16, img_names_jpg, output_names

def extract_prefix(filename):
    return filename.split('_')[0]

def convert_files(img_names_b16, img_names_jpg):
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

img_names_b16, img_names_jpg, output_names = get_img_names_from_folder(os.path.join('data', folder_name))

FWHM_grouped = defaultdict(list)

for i, img_name_b16 in enumerate(img_names_b16):
    img_path_b16 = os.path.join(folder_name, img_name_b16)
    img_path_jpg = os.path.join(folder_name, img_names_jpg[i])
    output_path = os.path.join('output', folder_name, output_names[i])
    
    sum, x = get_spectrual_line(output_path)
    x1, x2, FWHM_ = FWHM(x, sum)
    prefix = extract_prefix(img_name_b16)
    FWHM_grouped[prefix].append(FWHM_)

prefixes = []
FWHM_means = []
FWHM_errors = []

for prefix, FWHMs in FWHM_grouped.items():
    prefixes.append(float(prefix))
    FWHM_means.append(np.mean(FWHMs))
    FWHM_errors.append(np.std(FWHMs))

sorted_indices = np.argsort(prefixes)
prefixes = np.array(prefixes)[sorted_indices]
FWHM_means = np.array(FWHM_means)[sorted_indices]
FWHM_errors = np.array(FWHM_errors)[sorted_indices]

popt, pcov = curve_fit(quadratic, prefixes, FWHM_means, sigma=FWHM_errors)
a, b, c = popt

x_min = - b / (2 * a)
y_min = quadratic(x_min, a, b, c)
print(f'quadratic minimum point: x = {x_min}, y = {y_min}')

popt_g, pcov = curve_fit(gaussian, prefixes, FWHM_means, sigma=FWHM_errors, p0=[-10, 4, 1, 10])  # Initial guesses
a, x0, sigma, c = popt_g
y_min_g = gaussian(x0, a, x0, sigma, c)
print(f"Gaussian Minimum point: x = {x0}, y = {y_min_g}")

x_fit = np.linspace(min(prefixes), max(prefixes), 500)
y_fit = quadratic(x_fit, *popt)
y_fit_g = gaussian(x_fit, *popt_g)

plt.errorbar(prefixes, FWHM_means, yerr=FWHM_errors, fmt='o', capsize=5, label='FWHM data point')
plt.plot(x_fit, y_fit, label='Quadratic Fit')
plt.scatter(x_min, y_min, label=f"Quadratic Minimum (x={x_min:.2f}, y={y_min:.2f})")
plt.plot(x_fit, y_fit_g, label='Gaussian Fit')
plt.scatter(x0, y_min_g, label=f"Gaussian Minimum (x={x0:.2f}, y={y_min_g:.2f})")
plt.xlabel('Rotation (#)')
plt.ylabel('FWHM (pixel number)')
plt.legend()
plt.grid()
plt.show()
