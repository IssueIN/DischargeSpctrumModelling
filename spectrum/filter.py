import os
import numpy as np
import matplotlib.pyplot as plt
from pco_tools import pco_reader as pco
from scipy.ndimage import rotate
import json

# from utils import read_pco_img
def read_pco_img(img_path):
    img = pco.load(img_path)
    return img

def show_histogram(array, bins=50):
    """
    Displays a histogram of the values in a 2D array.
    
    Parameters:
    - array: 2D numpy array to analyze.
    - bins: Number of bins in the histogram.
    """
    # Flatten the 2D array into 1D for the histogram
    flattened_array = array.flatten()
    
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flattened_array, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    
    # Add titles and labels
    plt.title("Histogram of 2D Array Values", fontsize=16)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def average_col(grid):
    num_cols = grid.shape[1]
    avg_cols = np.zeros(num_cols)

    for i in range(num_cols):
        avg_cols[i] = grid[:, i].mean()
    
    return avg_cols

def has_spectral_lines(img, noise_threshold=20, line_intensity_threshold=415, coverage_threshold=0.02):
    cleaned_img = np.where(img > noise_threshold, img, 0)
    spectral_pixels = np.sum(cleaned_img > line_intensity_threshold)

    total_pixels = img.size

    return spectral_pixels / total_pixels > coverage_threshold

def classify_images(folder_path):
    spectral_images = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".b16"):
            img_path = os.path.join(folder_path, filename)
            img = read_pco_img(img_path)
            
            if has_spectral_lines(img):
                spectral_images.append(filename)
    
    return spectral_images

def rotate_image_grid(array, angle):
    rotated_array = rotate(array, angle, reshape=False, mode = 'nearest')
    return rotated_array

def crop_image(array, top_left, bottom_right):
    row_start, col_start = top_left
    row_end, col_end = bottom_right
    return array[row_start:row_end, col_start:col_end]

def display_array_as_image(array, cmap="gray"):
    """
    Displays a 2D array as an image.
    
    Parameters:
    - array: 2D numpy array to display.
    - cmap: Colormap for visualizing the array as an image (default is "gray").
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(array, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Intensity")
    plt.title("2D Array as Image")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # folder_path = input("Enter the folder path containing .pco files: ")
    folder_path = os.path.join('data', '21_11_plasma_first_attempt')
    spectral_images = classify_images(folder_path)
    
    print("Images with spectral lines:")
    for img_name in spectral_images:
        print(img_name)

    # folder_path = os.path.join('data', '29_11')
    # img_path = os.path.join(folder_path, 'take4_00068.b16')

    # with open(os.path.join(folder_path, 'config.json'), "r") as f:
    #     config = json.load(f)  
    # wavelengths = np.array(config["wavelengths"])
    # rotation_angle = config["rotation_angle"]
    # top_left = tuple(config["top_left"])
    # bottom_right = tuple(config["bottom_right"])

    # img = read_pco_img(img_path)
    # img = rotate_image_grid(img, rotation_angle)
    # noise_part = crop_image(img, (1500, 50), (2000, 2000))
    # sum_noi = average_col(noise_part)
    # img = crop_image(img, top_left, bottom_right)
    # sum = average_col(img)
    # sum = sum - sum_noi
    # sum = np.flip(sum)

    # # x_vals = np.arange(len(sum))
    # show_histogram(img)
    # plt.plot(wavelengths, sum)
    # plt.xlabel('wavelength (nm))')
    # plt.ylabel('spectral intensity (a. u.)')
    # plt.grid()
    # plt.show()
    # display_array_as_image(img)