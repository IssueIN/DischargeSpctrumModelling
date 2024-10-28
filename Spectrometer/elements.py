"""The module for classes representing optical elements"""


import numpy as np
from .physics import reflect

class OpticalElement():
    """
    The base class for general optical element.
    """

    def intercept(self, ray):
        """
        Calculate the intercept between the optical elements and a ray.

        Args:
            ray (Ray): The ray that propagates through the optical element.

        raises:
            NotImplementedError: if the intercept function is not implemented.
        """
        raise NotImplementedError(
            'intercept() needs to be implemented in derived classes')

    def propagate_ray(self, ray):
        """
        Propagate the ray through the optical element.

        Args:
            ray (Ray): The ray that propagates through the optical element.

        Raises:
            NotImplementedError: if the propagate_ray function is not implemented.
        """
        raise NotImplementedError(
            'propagate_ray() needs to be implemented in derived classes')
    

class OpticalSurfaceBase(OpticalElement):
        """
    Base class for optical surfaces with common functionalities.

    Attributes:
        __z_0 (float): The z-coordinate of the center of the spherical surface.
        __aper (float): The radius fo the aperture of the spherical surface.
        __curv (float): The curvature of the spherical surface.
        __n_1 (float): The refractive index of the medium outside the surface.
        __n_2 (float): The refractive index of the medium inside the surface.
    """