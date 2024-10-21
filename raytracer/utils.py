"""A module for utility functions used in ray tracing and related calculations."""

import numpy as np
import os


def norm(vector):
    """
    A function for normalizing a vector.

    Args:
        vector (list of numpy.ndarray): A list or array representing a vector.

    Returns:
        numpy.ndarray: An array represnting the normalized vector.
    """
    vector = vector / np.linalg.norm(vector)
    return vector


def plot_ray_series(ax, ray_series):
    """
    Recursively plot the trajectory of the RaySeries and its branches.

    Args:
        ax: The 3D axis to plot on.
        ray_series (RaySeries): The ray series object.
    """

    vertices = np.array(ray_series.vertices())
    ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2],
            label=f"Main, Intensity: {ray_series.intensity():.2f}")

    for branch in ray_series.get_branches():
        vertices = np.array(branch.vertices())
        ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                label=f"Branch, Intensity: {branch.intensity():.2f}")


def savefig(fig, name):
    """
    A function used to save figure to the output folder.

    Args:
        fig (figure): The figure object that need to be saved.
        name (string): the name of the figure file
    """
    output_dir = 'raytracer/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = name + '.png'

    fig.savefig(os.path.join(output_dir, filename))


def piecewise_linear_fit(x, slope, peak_x, peak_y):
    """
    The fit function for finding the focal point.

    Args:
        x (array-like or float): The x value(s) of the function.
        slope (float): the slope of the sides.
        peak_x (float): The x-coordinate of the turning point
        peak_y (float): The y_coordinate of the turning point.

    Returns:
        float or numpy array: The y value(s) corresponding to the given x.
    """
    if isinstance(x, (list, np.ndarray)):
        x = np.array(x)
        y = np.where(x < peak_x,
                     peak_y + slope * (peak_x - x),
                     peak_y + slope * (x - peak_x))
    else:
        if x < peak_x:
            y = peak_y + slope * (peak_x - x)
        else:
            y = peak_y + slope * (x - peak_x)

    return y


def quad_fit(x, a, x_peak, y_peak):
    """
    The quadratic fit function for finding the focal length

    Args:
        x (float or array-like): The x-coordinate(s) at which to evaluate the function.
        a (float): The curvature coefficient of the parabola. Determines how steep or shallow
            the parabola is.
        x_peak (float): The x-coordinate of the vertex of the parabola.
        y_peak (float): The y-coordinate of the vertex of the parabola.

    Returns:
        float or numpy.ndarray: The y value(s) corresponding to the given x according to the quadratic equation.
    """
    return a * (x - x_peak) ** 2 + y_peak
