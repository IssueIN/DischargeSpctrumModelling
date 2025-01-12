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

def wavelength_to_color(wavelength):
    if 380 <= wavelength < 450:
        return 'violet'
    elif 450 <= wavelength < 495:
        return 'blue'
    elif 495 <= wavelength < 520:
        return 'cyan'
    elif 520 <= wavelength < 565:
        return 'green'
    elif 565 <= wavelength < 590:
        return 'yellow'
    elif 590 <= wavelength < 620:
        return 'orange'
    elif 620 <= wavelength <= 750:
        return 'red'
    else:
        return 'gray'