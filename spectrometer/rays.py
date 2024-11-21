"""
A module for classes representing optical rays and bundles of rays.
"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import norm


class Ray:
    """
    A class representing optical rays, encapsulating their direction, position, and vertices.
    """

    def __init__(self, pos=[0., 0., 0.], direc=[0., 0., 1.], wavelength = None):
        """
        Args:
            pos (list or numpy.ndarray): A list or array represnting the position of the ray, default as [0., 0., 0.]
            direc (list or numpy.ndarray): A list or array represnting the direction vector of the array, default as [0., 0., 1.]
            wavelength (float): A float number representing the wavelength of the ray.

        Raises:
            ValueError: If `pos` does not contain exactly three elements.
            ValueError: If `direc` does no contain exactly three elements.
        """
        if len(pos) != 3:
            raise ValueError('Wrong size of position')
        if len(direc) != 3:
            raise ValueError('Wrong size of direction')

        direc = norm(np.array(direc))
        if wavelength is not None:
            wavelength = float(wavelength)

        self.__pos = np.array(pos)
        self.__direc = direc
        self.__vert = [self.__pos]
        self.__wl = wavelength

    def pos(self):
        """
        Returns:
            numpy.ndarray: A 3-elements array representing the x, y, z position of the ray
        """
        return self.__pos

    def direc(self):
        """
        Returns:
            numpy.ndarray: A 3-elements array representing the direction vector of the ray.
        """
        return self.__direc

    def vertices(self):
        """
        Returns:
            list: A list consisting of the vertices that the ray passed through.
        """
        return self.__vert
    
    def wavelength(self):
        """
        Returns:
            float: A float number representing the wavelength of the ray.
        """
        return self.__wl
    
    def set_wavelength(self, wl):
        """
        Set the wavelength for the ray object.
        """
        self.__wl = float(wl)


    def append(self, pos, direc):
        """
        Set the new position and directions, and append the new position to the vertices list.

        Args:
            pos (list): A list of three floats representing the new position of the ray.
            direc (list): A list of three floats representig the new direction of the ray.

        Raises:
            ValueError: If `pos` does not contain exactly three elements.
            ValueError: If `direc` does not contain exactly three elements.
        """
        if len(pos) != 3:
            raise ValueError('Wrong size of position')
        if len(direc) != 3:
            raise ValueError('Wrong size of direction')
        self.__pos = np.array(pos)
        self.__direc = norm(np.array(direc))
        self.__vert.append(self.__pos)

class RayBundle:
    """
    A class representing a bundle of rays
    """

    def __init__(self, rmax=5., nrings=5., multi=6, wavelength=500):
        ray_bundle = []
        for output in self._genpolar(rmax, nrings, multi):
            x = output[0] * np.cos(output[1])
            y = output[0] * np.sin(output[1])
            ray = Ray([x, y, 0], [0, 0, 1], wavelength=wavelength)
            ray_bundle.append(ray)

        self.__ray_bundle = ray_bundle
        self.__wavelength = wavelength
    
    def wavelength(self):
        return self.__wavelength

    def ray_bundle(self):
        """
        Returns:
            list: A list consisting of the ray objects in the ray bundle.
        """
        return self.__ray_bundle

    def _genpolar(self, rmax, nrings, multi):
        """
        Generates xy coordinates of points on concentric rings.

        Args:
            rmax (float): the radius of the outmost circle in the set of concentric circles.
            nrings (int): the number of rings.
            multi (int): the number of points in the innest rings (except the origin).

        Yields:
            tuple: A tuple contaning the radius and angle (r, theta) of a point.
        """
        yield (0, 0)
        for index, r in enumerate(np.linspace(0, rmax, num=int(nrings) + 1)):
            if r == 0:
                continue
            for theta in np.linspace(
                    0,
                    2 * np.pi,
                    num=index * multi,
                    endpoint=False):
                yield (r, theta)

    def propagate_bundle(self, elements, m=0):
        """
        Propagate the ray bundle through optical elements.

        Args:
            elements (list): A list of optical elements to propagate the rays through.
        """
        # for element in elements:
        #     for ray in self.ray_bundle():
        #         element.propagate_ray(ray)
        
        for element in elements:
            for ray in self.ray_bundle():
                if hasattr(element, 'propagate_ray') and 'm' in element.propagate_ray.__code__.co_varnames:
                    element.propagate_ray(ray, m=m)
                else:
                    element.propagate_ray(ray)
