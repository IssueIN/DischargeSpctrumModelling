"""
A module for classes representing lenses, including plano-convex and convex-plano lenses.
"""

from raytracer.elements import SphericalRefraction, OpticalElement


class Lens(OpticalElement):
    """
    A class representing a general lens, consisting of two curved surfaces.
    """

    def __init__(
            self,
            z_0,
            curvature1,
            curvature2,
            n_inside,
            n_outside,
            thickness,
            aperture):
        self.__d = thickness
        self.__z_0 = z_0
        self.__n_0 = n_outside
        self.__n_1 = n_inside

        sr1 = SphericalRefraction(
            z_0, aperture, curvature1, n_outside, n_inside)
        sr2 = SphericalRefraction(
            z_0 + thickness,
            aperture,
            curvature2,
            n_inside,
            n_outside)
        self.__srs = [sr1, sr2]

    def z_0(self):
        """
        Returns:
            float: The z-coordinate of the first surface of the lens.
        """
        return self.__z_0

    def thickness(self):
        """
        Returns:
            float: The thickness of the lens.
        """
        return self.__d

    def n_0(self):
        """
        Returns:
            float: The refractive index outside the lens.
        """
        return self.__n_0

    def n_1(self):
        """
        Returns:
            float: The refractive index inside the lens.
        """
        return self.__n_1

    def surfaces(self):
        """
        Returns:
            list: A 2-element list consisting of two surfaces.
        """
        return self.__srs

    def intercept(self, ray):
        """
        Determine the intercepts of the ray with the lens.

        Args:
            ray (Ray): The ray object, consisting of position, direction, and vertices.

        Returns:
            list: A list consist of two 3-element arrays representing x, y, z coordinates of the intercept points on each surface.
        """
        intercepts = []
        for sr in self.surfaces():
            intercepts.append(sr.intercept(ray))
            sr.propagate_ray(ray)
        return intercepts

    def propagate_ray(self, ray):
        """
        propagate the ray trough the two surfaces of the lens.

        Args:
            ray (Ray): The ray to be propagated through the lens.
        """
        for sr in self.surfaces():
            sr.propagate_ray(ray)

    def focal_point(self):
        """
        Calcuate the focal point of lens.

        Raises:
            ValueError: If the curvature of either surface is zero.

        Returns:
            float: the z-coordinate of the focal point of the lens.
        """
        d = self.thickness()
        n_diff = self.n_1() - self.n_0()

        try:
            r1 = 1 / self.surfaces()[0].curvature()
            r2 = 1 / self.surfaces()[1].curvature()
        except ZeroDivisionError as exc:
            raise ValueError('curvature cannot be zero') from exc

        _f = n_diff * (1 / r1 - 1 / r2 + d * n_diff / (self.__n_1 * r1 * r2))
        f = 1 / _f
        z_f = self.__z_0 + f
        return z_f


class PlanoConvex(Lens):
    """
    A class representing plano-convex lens.
    """

    def __init__(
            self,
            z_0,
            curvature,
            n_inside,
            n_outside,
            thickness,
            aperture):
        super().__init__(z_0, 0, curvature, n_inside, n_outside, thickness, aperture)

    def focal_point(self):
        """
        Calcuate the focal point of plano-convex lens.

        Returns:
            float: the z-coordinate of the focal point of the plano-convex lens.
        """
        d = self.thickness()
        n_diff = self.n_1() - self.n_0()
        r = 1 / self.surfaces()[1].curvature()

        fl = -r / n_diff + d
        z_f = self.z_0() + fl
        return z_f


class ConvexPlano(Lens):
    """
    A class representing convex-plano lens
    """

    def __init__(
            self,
            z_0,
            curvature,
            n_inside,
            n_outside,
            thickness,
            aperture):
        super().__init__(z_0, curvature, 0, n_inside, n_outside, thickness, aperture)

    def focal_point(self):
        """
        Calcuate the focal point of convex-plano lens

        Returns:
            float: the z-coordinate of the focal point of the convex-plano lens.
        """
        d = self.thickness()
        n_diff = self.n_1() - self.n_0()
        r = 1 / self.surfaces()[0].curvature()

        fl = r / n_diff + d * n_diff / self.n_1()
        z_f = self.z_0() + fl
        return z_f
