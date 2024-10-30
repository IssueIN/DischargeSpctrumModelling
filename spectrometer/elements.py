"""The module for classes representing optical elements"""


import numpy as np
from .physics import reflect, grating_equation
from .utils import norm

class OpticalElement():
    """
    The base class for general optical element.
    """

    def intercept(self, ray):
        """
        Calculate the intercept betwseen the optical elements and a ray.

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

    def __init__(self, pos, normal, aperture, curvature, n_1, n_2):
        if len(pos) != 3:
            raise ValueError('Wrong size of position')
        if len(normal) != 3:
            raise ValueError('Wrong size of direction')

        normal = norm(np.array(normal))
        self.__pos = np.array(pos)
        self.__norm = normal
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

    def normal(self):
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
        norm_p = self.normal()

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
    

class PlanarReflection(OpticalSurfaceBase):
    """
    A class for planar reflection, a special case of spherical refraction with zero curvature.
    """

    def __init__(self, pos, normal, aperture, n_1, n_2):
        super().__init__(pos=pos, normal=normal, aperture=aperture, curvature=0, n_1=n_1, n_2=n_2)
    
    def focal_point(self):
        return None
    
    def propagate_ray(self, ray):
        """
        Propagate the ray through the spherical surface, modelling reflection.

        Args:
            ray (Ray): the ray object, consisting position, direction, and vertices.

        Raises:
            ValueError: If no valid intercept is found.
            ValueError: if no valid relection.
        """
        intercept = self.intercept(ray)
        if intercept is None:
            raise ValueError("No valid intercept found for the ray.")

        reflec = reflect(ray.direc(), self.normal())
        if reflec is None:
            raise ValueError(" No valid rlected ray")

        ray.append(intercept, reflec)
    
class PlanarDiffractionGrating(PlanarReflection):
    """
    A class representing a planar diffraction grating, inheriting from PlanarReflection.

    Attributes:
        pos (list or np.ndarray): Position of the grating in 3D space.
        normal (list or np.ndarray): Normal vector of the grating surface.
        aperture (float): Aperture radius of the grating.
        n_1 (float): Refractive index outside the surface.
        n_2 (float): Refractive index inside the surface.
        rho_groove (float): Groove density of the diffraction grating (grooves per mm).
    """
    def __init__(self, pos, normal, aperture, n_1, n_2, rho_groove):
        super().__init__(pos=pos, normal=normal, aperture=aperture, n_1=n_1, n_2=n_2)
        self.__rho_groove = rho_groove
    
    def grating_spacing(self):
        """
        Returns:
            float: Calculate and return the grating spacing based on groove density.
        """
        return 1.0 / (self.__rho_groove * 1000)
    
    def propagate_ray(self, ray, m=1):
        """
        Propagate the ray through the diffraction grating, modelling reflection.

        Args:
            ray (Ray): the ray object, consisting position, direction, wavelength, and vertices.
            m (Integer): the interested diffraction order. 

        Raises:
            ValueError: If no valid intercept is found.
            ValueError: if no valid diffraction.
        """
        intercept = self.intercept(ray)
        if intercept is None:
            raise ValueError("No valid intercept found for the ray.")
        
        direc_r = ray.direc()
        normal = self.normal()
        d = self.grating_spacing()
        wl = ray.wavelength()
        
        diffrac = grating_equation(direc_r, normal, m, d, wl)
        if diffrac is None:
            raise ValueError(" No valid diffracted ray")
        
        ray.append(intercept, diffrac)

class OutputPlane(OpticalSurfaceBase):
    """
    the class representing the output plane in an optical system.

    The output plane records the intercept point of a ray without changing its direction.
    """

    def __init__(self, pos, normal):
        super().__init__(pos=pos, normal=normal, aperture=float('inf'), curvature=0, n_1=None, n_2=None)

    def propagate_ray(self, ray):
        """
        Propagate the ray to the output plane.

        Args:
            ray (Ray): the ray object, consisting of its position and direction.
        """
        intercept = self.intercept(ray)
        if intercept is not None:
            ray.append(intercept, ray.direc())
