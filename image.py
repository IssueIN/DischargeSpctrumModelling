from spectrum.utils import read_pco_img, rotate_image_grid, crop_image
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
    fig.colorbar(cax, ax=ax, label="Intensity", orientation='horizontal')  # Add colorbar
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")
    fig.tight_layout()
    return fig


folder_path = os.path.join('data', '28_11')
file_name = 'take4_00136.b16'
file_path = os.path.join(folder_path, file_name)


with open(os.path.join(folder_path, 'config.json'), "r") as f:
        config = json.load(f)  
wavelengths = np.array(config["wavelengths"])
rotation_angle = config["rotation_angle"]
top_left = tuple(config["top_left"])
bottom_right = tuple(config["bottom_right"])

img = read_pco_img(file_path)
img = rotate_image_grid(img, rotation_angle)
img = crop_image(img, top_left, bottom_right)

fig = display_array_as_image(img)

plt.savefig(os.path.join('fig', 'test.png'), dpi=300)
plt.show()