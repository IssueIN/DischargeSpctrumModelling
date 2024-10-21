"""
A module for classes representing optical rays and bundles of rays.
"""

import numpy as np
import matplotlib.pyplot as plt
from raytracer.utils import norm


class Ray:
    """
    A class representing optical rays, encapsulating their direction, position, and vertices.
    """

    def __init__(self, pos=[0., 0., 0.], direc=[0., 0., 1.]):
        """
        Args:
            pos (list or numpy.ndarray): A list or array represnting the position of the ray, default as [0., 0., 0.]
            direc (list or numpy.ndarray): A list or array represnting the direction vector of the array, default as [0., 0., 1.]

        Raises:
            ValueError: If `pos` does not contain exactly three elements.
            ValueError: If `direc` does no contain exactly three elements.
        """
        if len(pos) != 3:
            raise ValueError('Wrong size of position')
        if len(direc) != 3:
            raise ValueError('Wrong size of direction')

        direc = norm(np.array(direc))

        self.__pos = np.array(pos)
        self.__direc = direc
        self.__vert = [self.__pos]

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

    def __init__(self, rmax=5., nrings=5., multi=6):
        ray_bundle = []
        for output in self._genpolar(rmax, nrings, multi):
            x = output[0] * np.cos(output[1])
            y = output[0] * np.sin(output[1])
            ray = Ray([x, y, 0], [0, 0, 1])
            ray_bundle.append(ray)

        self.__ray_bundle = ray_bundle

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

    def propagate_bundle(self, elements):
        """
        Propagate the ray bundle through optical elements.

        Args:
            elements (list): A list of optical elements to propagate the rays through.
        """
        for element in elements:
            for ray in self.ray_bundle():
                element.propagate_ray(ray)

    def track_plot(self):
        """
        Plot the track of the ray bundle.

        Returns:
            matlibplot.figure.Figure: A 3D plot showing the trajectory of the ray bundle.
        """
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        for ray in self.ray_bundle():
            verts = np.array(ray.vertices())
            ax.plot3D(verts[:, 0], verts[:, 1], verts[:, 2])

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        return fig

    def rms(self):
        """
        Calculate the root-mean-square (RMS) of the ray bundle positions on the xy plane.

        returns:
            float: the RMS value of the ray bundle positions.
        """
        pos = np.array([ray.pos() for ray in self.ray_bundle()])
        r_sq = np.sum(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        r_rms = np.sqrt(r_sq / len(self.ray_bundle()))
        return r_rms

    def spot_plot(self):
        """
        Plot the positions of rays in the bundle on the xy plane

        returns:
            matplotlib.figure.Figure: the xy plot of rays' positions.
        """
        fig = plt.figure()

        for ray in self.ray_bundle():
            pos = ray.pos()
            plt.scatter(pos[0], pos[1], color='blue', s=8)

        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.grid()
        # set the aspect ratio equal
        plt.gca().set_aspect('equal', adjustable='box')

        return fig


class RaySeries(Ray):
    """
    A class for rays with multiple branches.
    """

    def __init__(self, pos=[0., 0., 0.], direc=[0., 0., 1.], intensity=1.):
        if len(pos) != 3:
            raise ValueError('Wrong size of position')
        if len(direc) != 3:
            raise ValueError('Wrong size of direction')

        direc = norm(np.array(direc))

        self.__pos = np.array(pos)
        self.__direc = direc
        self._intensity = intensity
        self.__vert = [np.append(self.pos(), intensity)]
        self._branches = []  # list for reflected branches

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

    def intensity(self):
        """
        Returns:
            float: the intensity of curret ray.
        """
        return self._intensity

    def vertices(self):
        """
        Returns:
            list: A list consisting of the vertices that the ray passed through.
        """
        return self.__vert

    def get_branches(self):
        """
        Returns:
            list: A list containing all RaySeries objects that representing the reflected branches of the ray.
        """
        return self._branches

    def add_branch(self, pos, direc, intensity):
        """
        Create new branch, and append it to the branches list.

        Args:
            pos (list): A list of three floats representing the new position of the branch.
            direc (list): A list of three floats representig the new direction of the branch.
            intensity (float): the intensity of the branch.

        Raises:
            ValueError: If `pos` does not contain exactly three elements.
            ValueError: If `direc` does not contain exactly three elements.
        """
        if len(pos) != 3:
            raise ValueError('Wrong size of position')
        if len(direc) != 3:
            raise ValueError('Wrong size of direction')
        direc = norm(np.array(direc))

        branch = RaySeries(pos=pos, direc=direc, intensity=intensity)
        self._branches.append(branch)

    def append(self, pos, direc, intensity=None):
        """
        Set the new position and directions, and append the new position to the vertices list.

        Args:
            pos (list): A list of three floats representing the new position of the ray.
            direc (list): A list of three floats representig the new direction of the ray.
            intensity (float): the new intensity of the ray

        Raises:
            ValueError: If `pos` does not contain exactly three elements.
            ValueError: If `direc` does not contain exactly three elements.
        """
        if len(pos) != 3:
            raise ValueError('Wrong size of position')
        if len(direc) != 3:
            raise ValueError('Wrong size of direction')
        if intensity is None:
            intensity = self.intensity()
        self.__pos = np.array(pos)
        self.__direc = norm(np.array(direc))
        self.__intensity = intensity
        self.__vert.append(np.append(self.__pos, intensity))
