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