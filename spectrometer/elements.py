"""The module for classes representing optical elements"""


import numpy as np
from .physics import reflect
from .utils import norm

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
        pos (list or numpy.ndarray): A list or array represnting the position of the surface.
        norm (list or numpy.ndarray): A list or array represnting the normal direction vector of the sruface.
        __aper (float): The radius fo the aperture of the spherical surface.
        __curv (float): The curvature of the spherical surface.
        __n_1 (float): The refractive index of the medium outside the surface.
        __n_2 (float): The refractive index of the medium inside the surface.
    """

    def __init__(self, pos, norm, aperture, curvature, n_1, n_2):
        if len(pos) != 3:
            raise ValueError('Wrong size of position')
        if len(norm) != 3:
            raise ValueError('Wrong size of direction')

        norm = norm(np.array(norm))
        self.__pos = np.array(pos)
        self.__norm = norm
        self.__aperture = aperture
        self.__curvature = curvature
        self.__n_1 = n_1
        self.__n_2 = n_2
    
    def pos(self):
        """
        Returns:
            numpy.ndarray: A 3-elements array representing the x, y, z position of the surface.
        """
        return self.__pos

    def norm(self):
        """
        Returns:
            numpy.ndarray: A 3-elements array representing the direction vector of the surface.
        """
        return self.__norm

    def aperture(self):
        """
        Returns:
            float: The radius of the aperture.
        """
        return self.__aperture

    def curvature(self):
        """
        Returns:
            float: The curvature of the surface.
        """
        return self.__curvature

    def n_1(self):
        """
        Returns:
            float: The refractive index outside the surface.
        """
        return self.__n_1

    def n_2(self):
        """
        Returns:
            float: The refractive index inside the surface.
        """
        return self.__n_2
    

class PlanarReflection(OpticalElement):
    """
    A class for planar reflection, a special case of spherical refraction with zero curvature.
    """

    def __init__(self, pos, norm, aperture, n_1, n_2):
        super().__init__(pos=pos, norm=norm, aperture=aperture, curvature=0, n_1=n_1, n_2=n_2)
    

    def intercept(self, ray):
        """
        Calculate the intercept point between the ray and the planar surface.

        Args:
            ray (Ray): The ray that propagates towards the optical element.

        Returns:
            numpy.ndarray or None: The 3D intercept point on the surface, or None if there is no intersection.
        """
        pos_r = ray.pos()
        direc_r = ray.direc()

        pos_p = self.pos()
        norm_p = self.norm()

        # Ray Equation: R(t) = P_r + t * D_r
        # Plane Equation: (R(t) - P_p) * N_p = 0
        # Solve for t, substitute t back into Ray equation

        denom = np.dot(direc_r, norm_p)
        if np.abs(denom) < 1e-6:
            return None
        
        t = np.dot(pos_p - pos_r, norm_p) / denom

        if t < 0:
            return None
        
        intercept = pos_r + t * direc_r

        distance =  np.linalg.norm(intercept - pos_p)

        if intercept is None or distance > self.aperture():
            return None
        
        return intercept
    
    def focal_point(self):
        return None
    
    def propagate_ray(self, ray):
        """
        Propagate the ray through the spherical surface, modelling reflection.

        Args:
            ray (Ray): the ray object, consisting position, direction, and vertices.

        Raises:
            ValueError: If no valid intercept is found
            ValueError: if no valid relection.
        """
        intercept = self.intercept(ray)
        if intercept is None:
            raise ValueError("No valid intercept found for the ray.")

        reflec = reflect(ray.direc(), self.norm(intercept))
        if reflec is None:
            raise ValueError(" No valid rlected ray")

        ray.append(intercept, reflec)
