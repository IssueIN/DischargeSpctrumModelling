from spectrum.utils import read_pco_img, rotate_image_grid, crop_image, sum_up_col
import matplotlib.pyplot as plt
import os
import json
import numpy as np

def display_array_as_image(array, cmap="gray"):
    """
    Displays a 2D array as an image.
    
    Parameters:
    - array: 2D numpy array to display.
    - cmap: Colormap for visualizing the array as an image (default is "gray").
    """
    fig, ax = plt.subplots(figsize=(7, 4))  # Create a figure and axis
    cax = ax.imshow(array, cmap=cmap, interpolation="nearest")  # Show image
    fig.colorbar(cax, ax=ax, label="Intensity", orientation = 'horizontal')  # Add colorbar
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")
    fig.tight_layout()
    return fig

def average_col(grid, row_range=None):
    """
    Computes the average intensity for each column over a specified range of rows.

    Parameters:
    - grid: 2D numpy array representing the image grid.
    - row_range: Tuple of two integers specifying the start and end rows to average (inclusive).
                 If None, averages over all rows.

    Returns:
    - avg_cols: 1D numpy array of average intensities for each column.
    """
    if row_range is None:
        row_range = (0, grid.shape[0])  # Default to all rows if range is not specified
    start_row, end_row = row_range
    avg_cols = np.mean(grid[start_row:end_row, :], axis=0)
    return avg_cols


folder_path = os.path.join('data', '28_11')
file_name = 'take3_00130.b16'
file_path = os.path.join(folder_path, file_name)


with open(os.path.join(folder_path, 'config.json'), "r") as f:
        config = json.load(f)  
wavelengths = np.array(config["wavelengths"])
rotation_angle = config["rotation_angle"]
top_left = tuple(config["top_left"])
bottom_right = tuple(config["bottom_right"])

# top_left = (200,0)
# bottom_right = (950,3200)

img = read_pco_img(file_path)
img = rotate_image_grid(img, rotation_angle)
# noise_part = crop_image(img, (bottom_right[0], top_left[1]), (bottom_right[0] + 100, bottom_right[1]))
img = crop_image(img, top_left, bottom_right)
row_range=(520,540)
avg = average_col(img, row_range)

# sum_noi = average_col(noise_part)

# avg = avg - sum_noi

avg = np.flip(avg)
x = np.arange(len(avg))

plt.figure()
fig, ax = plt.subplots(figsize=(7, 4))
plt.plot(x, avg, c='black', linewidth=0.6)
plt.xlabel('pixel number')
plt.ylabel('Spectral Intensity (a.u.)')
plt.grid()

plt.savefig(os.path.join('fig', 'sliced_spectrum_5'), dpi=300)
plt.show()
plt.close()