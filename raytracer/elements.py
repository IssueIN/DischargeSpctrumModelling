"""
The module for classes representing optical elements, including spherical refraction, spherical reflection, and output plane.
"""


import numpy as np
from raytracer.physics import refract, reflect, reflectivity


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

    def __init__(self, z_0, aperture, curvature, n_1, n_2):
        self.__z_0 = z_0
        self.__aperture = aperture
        self.__curvature = curvature
        self.__n_1 = n_1
        self.__n_2 = n_2

    def z_0(self):
        """
        Returns:
            float: The z-coordinate of the surface.
        """
        return self.__z_0

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
        Calculate the intercept of a ray with the spherical surface.

        Args:
            ray (Ray): The ray object, consisting of position, direction, and vertices.

        Returns:
            None: If the determinant is less than zero, resulting in an imaginary intercept.
            None: If the ray does not intersect the surface.
            numpy.ndarray: a 3-element array representing x, y, z coordinates of the intercept point.
        """
        pos = ray.pos()
        direc = ray.direc()

        if self.curvature() == 0:
            z_diff = self.z_0() - pos[2]
            l = z_diff / direc[2]
            if l < 0:
                return None
            intercept = pos + l * direc
        else:
            radius = 1 / self.curvature()
            origin = np.array([0., 0., self.z_0() + radius])
            r = pos - origin
            r_dot_k = np.inner(r, direc)
            r_sq = np.linalg.norm(r) ** 2
            det = r_dot_k ** 2 - (r_sq - radius ** 2)

            if det < 0:
                return None

            l_p = -r_dot_k + np.sqrt(det)
            l_m = -r_dot_k - np.sqrt(det)

            l = min(filter(lambda x: x > 0, [l_p, l_m]), default=None)
            if l is None:
                return None
            intercept = pos + l * direc

        if intercept is None or (
                intercept[0] ** 2 + intercept[1] ** 2) > (self.aperture() ** 2):
            return None

        return intercept

    def normal(self, intercept):
        """
        Determine the normal vector at the point of incidence on the surface.

        Args:
            intercept (numpy.ndarray): A 3-element array representing the x, y, z coordinates of the intercept.

        Returns:
            np.ndarray: A 3-element array representing the normal vector of the surface.
        """
        if self.curvature() == 0:
            return np.array([0., 0., -1.])
        else:
            radius = 1 / self.curvature()
            origin = np.array([0., 0., self.z_0() + radius])
            normal = intercept - origin
            return normal

    def plot(self, ax, label='Spherical Surface'):
        """
        Plot spherical surface in the 3D graph.

        Args:
            ax: The 3D axis to plot on.
            label (string): The label of the surface.
        """
        radius = 1 / self.curvature()

        theta, phi = np.linspace(
            0, 2 * np.pi, 100), np.linspace(0, np.pi / 2, 100)
        THETA, PHI = np.meshgrid(theta, phi)
        x = radius * np.sin(PHI) * np.cos(THETA)
        y = radius * np.sin(PHI) * np.sin(THETA)
        z = self.z_0() - radius * np.cos(PHI)

        mask = np.sqrt(x ** 2 + y ** 2) <= self.aperture()

        x = np.where(mask, x, np.nan)
        y = np.where(mask, y, np.nan)
        z = np.where(mask, z, np.nan)

        ax.plot_surface(x, y, z, alpha=0.5, label=label)


class SphericalRefraction(OpticalSurfaceBase):
    """
    A class for spherical refraction, modeling refraction through a spherical surface.
    """

    def __init__(self, z_0, aperture, curvature, n_1, n_2):
        super().__init__(z_0, aperture, curvature, n_1, n_2)

    def propagate_ray(self, ray):
        """
        Propagate the ray through the spherical surface, modelling refraction.

        Args:
            ray (Ray): the ray object, consisting position, direction, and vertices.

        Raises:
            ValueError: If no valid intercept is found
            ValueError: if total internal reflection occurs.
        """
        intercept = self.intercept(ray)
        if intercept is None:
            raise ValueError("No valid intercept found for the ray.")

        refrac = refract(
            ray.direc(),
            self.normal(intercept),
            self.n_1(),
            self.n_2())
        if refrac is None:
            raise ValueError(
                "Total internal reflection occured. No valid refracted ray")

        ray.append(intercept, refrac)

    def focal_point(self):
        """
        Calculate the focal point of the spherical surface.

        Returns:
            float: the z-coordinate of the focal point.
        """
        radius = 1 / self.curvature()
        fl = (self.n_2() / (self.n_2() - self.n_1())) * radius
        z_f = self.z_0() + fl
        return z_f


# class PlanarRefraction(SphericalRefraction):
#     """
#     A class for planar refraction, a special case of spherical refraction with zero curvature.
#     """

#     def __init__(self, z_0, aperture, n_1, n_2):
#         super().__init__(z_0=z_0, aperture=aperture, curvature=0, n_1=n_1, n_2=n_2)

#     def focal_point(self):
#         """
#         Calculate the focal point of the planar surface.

#         Returns:
#             None: Planar surfaces do not have a focal point
#         """
#         return None

class SphericalReflection(OpticalSurfaceBase):
    """
    A class for spherical reflection, modeling reflection through a spherical surface.
    """

    def __init__(self, z_0, aperture, curvature, n_1, n_2):
        super().__init__(z_0, aperture, curvature, n_1, n_2)

    def propagate_ray(self, ray):
        """
        Propagate the ray through the spherical surface, modelling reflection.

        Args:
            ray (Ray): the ray object, consisting position, direction, and vertices.

        Raises:
            ValueError: If no valid intercept is found
            ValueError: if no valid reflection ray is found.
        """
        intercept = self.intercept(ray)
        if intercept is None:
            raise ValueError("No valid intercept found for the ray.")

        reflec = reflect(ray.direc(), self.normal(intercept))

        if reflec is None:
            raise ValueError(
                "No valid reflected ray")

        ray.append(intercept, reflec)


class SphericalSurface(OpticalSurfaceBase):
    """
    A class for spherical surface.

    Attributes:
        __z_0 (float): The z-coordinate of the center of the spherical surface.
        __aper (float): The radius fo the aperture of the spherical surface.
        __curv (float): The curvature of the spherical surface.
        __n_1 (float): The refractive index of the medium outside the surface.
        __n_2 (float): The refractive index of the medium inside the surface.
    """

    def __init__(self, z_0, aperture, curvature, n_1, n_2):
        super().__init__(z_0, aperture, curvature, n_1, n_2)

    def propagate_raySeries(self, ray):
        """
        Propagate the ray series through the spherical surface, modeling both refraction and reflection.

        Args:
            ray (RaySeries): a ray series object representing the incident ray

        Raises:
            ValueError: if no valid refraction and reflection founded
        """
        indicator = False  # indicates if a new branch has been added
        intercept = self.intercept(ray)

        if intercept is None:
            for branch in ray.get_branches():
                self.propagate_raySeries(branch)
            return  # Skip further processing if no valid intercept

        normal = self.normal(intercept)
        direc = ray.direc()

        refrac = refract(direc, normal, self.n_1(), self.n_2())
        reflec = reflect(direc, normal)
        Reflectivity = reflectivity(direc, normal, self.n_1(), self.n_2())
        Transmissivity = 1 - Reflectivity

        if refrac is None and reflec is None:
            raise ValueError('No valid refraction and reflection founded')

        if refrac is None:
            ray.append(intercept, reflec, ray.intensity())
        elif reflec is None:
            ray.append(intercept, refrac, Transmissivity * ray.intensity())
        else:
            ray.append(intercept, refrac, Transmissivity * ray.intensity())
            ray.add_branch(intercept, reflec, Reflectivity * ray.intensity())
            indicator = True

        # recursively propagate through all the existed branches
        if indicator is True:
            for branch in ray.get_branches()[:-1]:
                self.propagate_raySeries(branch)
        else:
            for branch in ray.get_branches():
                self.propagate_raySeries(branch)


class OutputPlane(OpticalSurfaceBase):
    """
    the class representing the output plane in an optical system.

    The output plane records the intercept point of a ray without changing its direction.
    """

    def __init__(self, z_0):
        super().__init__(z_0, aperture=float('inf'), curvature=0, n_1=None, n_2=None)

    def propagate_ray(self, ray):
        """
        Propagate the ray to the output plane.

        Args:
            ray (Ray): the ray object, consisting of its position and direction.
        """
        intercept = self.intercept(ray)
        ray.append(intercept, ray.direc())

    def propagate_raySeries(self, raySeries):
        """
        Propagate the rayseries to the output plane.

        Args:
            raySeries (RaySeries): the raySeries object, consisting multiple branches.
        """
        intercept = self.intercept(raySeries)
        if intercept is not None:
            raySeries.append(intercept, raySeries.direc())

        for branch in raySeries.get_branches():
            self.propagate_raySeries(branch)
