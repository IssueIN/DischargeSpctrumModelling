"""The module for classes representing optical elements"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from spectrometer.physics import reflect, grating_equation
from spectrometer.utils import norm, wavelength_to_color
import inspect

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
        __pos (list or numpy.ndarray): A list or array represnting the position of the surface.
        __norm (list or numpy.ndarray): A list or array represnting the normal direction vector of the sruface.
        __aperture (float): The radius fo the aperture of the spherical surface.
        __curvature (float): The curvature of the spherical surface.
        __n_1 (float): The refractive index of the medium outside the surface.
        __n_2 (float): The refractive index of the medium inside the surface.
    """

    def __init__(self, pos, aperture, curvature, n_1, n_2):
        if len(pos) != 3:
            raise ValueError('Wrong size of position')

        self.__pos = np.array(pos)
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

class SphericalSurfaceBase(OpticalSurfaceBase):
    """
    Base class for optical spherical surfaces with common functionalities.

    Attributes:
        __pos (list or numpy.ndarray): A list or array represnting the position of the origin.
        __direc (list or numpy.ndarray): A list or array representing the direction of the surface.
        __aperture (float): The radius fo the aperture of the spherical surface.
        __curvature (float): The curvature of the spherical surface.
        __n_1 (float): The refractive index of the medium outside the surface.
        __n_2 (float): The refractive index of the medium inside the surface.
    """
    def __init__(self, pos, direc, aperture, curvature, n_1, n_2):
        super().__init__(pos=pos, aperture=aperture, curvature=curvature, n_1=n_1, n_2=n_2)
        if len(direc) != 3:
            raise ValueError('Wrong size of normal direction')

        direc = norm(np.array(direc))
        self.__direc = direc
        self.radius = abs(1 / curvature)
        self.is_concave = curvature < 0
    
    def direc(self):
        """
        Returns:
            numpy.ndarray: A 3-elements array representing the direction vector of the surface.
        """
        return self.__direc
        
    
    def intercept(self, ray):        
        """
        Calculate the intercept point between the ray and the spherical surface.

        Args:
            ray (Ray): The ray that propagates towards the optical element.

        Returns:
            numpy.ndarray or None: The 3D intercept point on the surface, or None if there is no intersection.
        """
        pos_r = ray.pos()
        direc_r = ray.direc()

        radius = 1 / self.curvature()
        origin = self.pos()
        r = pos_r - origin
        r_dot_k = np.inner(r, direc_r)
        r_sq = np.linalg.norm(r) ** 2
        det = r_dot_k ** 2 - (r_sq - radius ** 2)

        if det < 0:
            return None

        l_p = -r_dot_k + np.sqrt(det)
        l_m = -r_dot_k - np.sqrt(det)

        if self.is_concave:
            l = max(filter(lambda x: x > 0, [l_p, l_m]), default=None)
        else:
            l = min(filter(lambda x: x > 0, [l_p, l_m]), default=None)
            
        if l is None:
            return None

        intercept = pos_r + l * direc_r

        to_intercept = intercept - origin
        direc = self.direc()

        to_intercept_proj = to_intercept - np.dot(to_intercept, direc) * direc
        distance_sq = np.linalg.norm(to_intercept_proj) ** 2

        if intercept is None or (distance_sq) > (self.aperture() ** 2):
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
        origin = self.pos()
        normal = intercept - origin
        normal = norm(normal)
        return normal

class PlanarSurfaceBase(OpticalSurfaceBase):
    """
    Base class for optical planar surfaces with common functionalities.

    Attributes:
        __pos (list or numpy.ndarray): A list or array represnting the position of the surface.
        __norm (list or numpy.ndarray): A list or array represnting the normal direction vector of the sruface.
        __aperture (float): The radius fo the aperture of the spherical surface.
        __curvature (float): The curvature of the spherical surface.
        __n_1 (float): The refractive index of the medium outside the surface.
        __n_2 (float): The refractive index of the medium inside the surface.
    """
    def __init__(self, pos, normal, aperture, n_1, n_2):
        super().__init__(pos=pos, aperture=aperture, curvature=0, n_1=n_1, n_2=n_2)
        if len(normal) != 3:
            raise ValueError('Wrong size of normal direction')

        normal = norm(np.array(normal))
        self.__norm = normal
    
    def normal(self):
        """
        Returns:
            numpy.ndarray: A 3-elements array representing the direction vector of the surface.
        """
        return self.__norm
    
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
        if np.abs(denom) < 1e-8:
            return None
        
        t = np.dot(pos_p - pos_r, norm_p) / denom

        if t < 0:
            return None
        
        intercept = pos_r + t * direc_r

        distance =  np.linalg.norm(intercept - pos_p)

        if intercept is None or distance > self.aperture():
            print('aperture')
            return None
        
        return intercept
    
class DiffractionGratingBase(OpticalElement):
    """
    Base class for diffraction gratings with common functionalities.

    Attributes:
        __rho_groove (float): Groove density of the diffraction grating (grooves per mm).
    """
    def __init__(self, rho_groove):
        self.__rho_groove = rho_groove

    def grating_spacing(self):
        """
        Calculate and return the grating spacing based on groove density.

        Returns:
            float: The spacing between the grating lines.
        """
        return 1.0 / (self.__rho_groove * 1000)
    
    def normal(self):
        """
        Calculate the normal vector of the diffraction grating.

        raises:
            NotImplementedError: if the normal function is not implemented.
        """
        raise NotImplementedError(
            'normal() needs to be implemented in derived classes')
    
    def propagate_ray(self, ray, m=0):
        """
        Propagate the ray through the diffraction grating, modelling diffraction.

        Args:
            ray (Ray): The ray object, consisting of position, direction, wavelength, and vertices.
            m (int): The diffraction order of interest.

        Raises:
            ValueError: If no valid intercept or diffraction is found.
        """
        intercept = self.intercept(ray)
        if intercept is None:
            raise ValueError("No valid intercept found for the ray.")
        
        direc_r = ray.direc()
        if 'intercept' in inspect.signature(self.normal).parameters:
            normal = self.normal(intercept)  # Call with intercept if needed
        else:
            normal = self.normal()  # Call without intercept
        d = self.grating_spacing()
        wl = ray.wavelength()

        diffrac = grating_equation(direc_r, normal, m, d, wl)
        if diffrac is None:
            raise ValueError("No valid diffracted ray found.")
        
        ray.append(intercept, diffrac)

class SphericalReflection(SphericalSurfaceBase):
    """
    A class for spherical reflection, modeling reflection through a spherical surface.
    """

    def __init__(self, pos, direc, aperture, curvature, n_1, n_2):
        super().__init__(pos=pos, direc=direc, aperture=aperture, curvature=curvature, n_1=n_1, n_2=n_2)

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

class SphericalDiffractionGrating(SphericalSurfaceBase):
    """
    A class representing a spherical diffraction grating, inheriting from PlanarReflection.

    Attributes:
        pos (list or np.ndarray): Position of the grating in 3D space.
        normal (list or np.ndarray): Normal vector of the grating surface.
        aperture (float): Aperture radius of the grating.
        n_1 (float): Refractive index outside the surface.
        n_2 (float): Refractive index inside the surface.
        rho_groove (float): Groove density of the diffraction grating (grooves per mm).
    """
    def __init__(self, pos, direc, aperture, curvature, rho_groove, n_1=1.0, n_2=1.5):
        super().__init__(pos=pos, direc=direc, aperture=aperture, curvature=curvature, n_1=n_1, n_2=n_2)
        self.__rho_groove = rho_groove
    
    def grating_spacing(self):
        """
        Calculate and return the grating spacing based on groove density.

        Returns:
            float: The spacing between the grating lines.
        """
        return 1.0 / (self.__rho_groove * 1000)

    
    def propagate_ray(self, ray, m=0):
        """
        Propagate the ray through the diffraction grating, modelling diffraction.

        Args:
            ray (Ray): The ray object, consisting of position, direction, wavelength, and vertices.
            m (int): The diffraction order of interest.

        Raises:
            ValueError: If no valid intercept or diffraction is found.
        """
        intercept = self.intercept(ray)
        if intercept is None:
            raise ValueError("No valid intercept found for the ray.")
        
        direc_r = ray.direc()
        normal = self.normal(intercept)
        d = self.grating_spacing()
        wl = ray.wavelength()

        diffrac = grating_equation(direc_r, normal, m, d, wl)
        if diffrac is None:
            raise ValueError("No valid diffracted ray found.")
        
        ray.append(intercept, diffrac)
    


class PlanarReflection(PlanarSurfaceBase):
    """
    A class for planar reflection, a special case of spherical refraction with zero curvature.
    """

    def __init__(self, pos, normal, aperture, n_1=1.0, n_2=1.5):
        super().__init__(pos=pos, normal=normal, aperture=aperture, n_1=n_1, n_2=n_2)
    
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
    
class PlanarDiffractionGrating(PlanarSurfaceBase, DiffractionGratingBase):
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
    def __init__(self, pos, normal, aperture, rho_groove, n_1=1.0, n_2=1.5):
        super().__init__(pos=pos, normal=normal, aperture=aperture, n_1=n_1, n_2=n_2)
        self.__rho_groove = rho_groove
    
    def grating_spacing(self):
        """
        Calculate and return the grating spacing based on groove density.

        Returns:
            float: The spacing between the grating lines.
        """
        return 1.0 / (self.__rho_groove * 1000)

    
    def propagate_ray(self, ray, m=0):
        """
        Propagate the ray through the diffraction grating, modelling diffraction.

        Args:
            ray (Ray): The ray object, consisting of position, direction, wavelength, and vertices.
            m (int): The diffraction order of interest.

        Raises:
            ValueError: If no valid intercept or diffraction is found.
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
            raise ValueError("No valid diffracted ray found.")
        
        ray.append(intercept, diffrac)


class OutputPlane(PlanarSurfaceBase):
    """
    the class representing the output plane in an optical system.

    The output plane records the intercept point of a ray without changing its direction.
    """

    def __init__(self, pos, normal):
        super().__init__(pos=pos, normal=normal, aperture=float('inf'), n_1=None, n_2=None)
        self.aggregated_intercepts = []
        self.wavelengths = []

    def propagate_ray(self, ray):
        """
        Propagate the ray to the output plane.

        Args:
            ray (Ray): the ray object, consisting of its position and direction.
        """
        intercept = self.intercept(ray)

        if intercept is not None:
            ray.append(intercept, ray.direc())
            self.aggregated_intercepts.append(intercept)
            self.wavelengths.append(ray.wavelength())

    def scatter_plot(self):
        # intercepts = np.array(self.intercepts)

        # fig = plt.figure(figsize=(8, 6))
        # plt.scatter(intercepts[:, 1], intercepts[:, 2],
        #                       c=[wavelength_to_color(wl * 1e9) for wl in self.wavelengths], marker='o')

        # plt.title("Intercepts on the Output Plane")
        # plt.xlabel("Y-axis (m)")
        # plt.ylabel("Z-axis (m)")
        
        # plt.grid(True)

        # return fig
        intercepts = np.array(self.aggregated_intercepts)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        # Scatter plot of intercepts with wavelength as color
        scatter = ax.scatter(intercepts[:, 1], intercepts[:, 2],
                             c=[wavelength_to_color(wl * 1e9) for wl in self.wavelengths], marker='o', s=6)
        
        # Customizing the plot
        ax.set_title("Intercepts on the Output Plane")
        ax.set_xlabel("Y-axis (m)")
        ax.set_ylabel("Z-axis (m)")

        # Adding the color legend manually
        handles = []
        labels = []
        unique_wavelengths = sorted(set(self.wavelengths))  # Ensure unique wavelengths
        for wl in unique_wavelengths:
            color = wavelength_to_color(wl * 1e9)  # Convert wavelength to color
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
            labels.append(f'{wl*1e9:.0f} nm')

        ax.legend(handles=handles, labels=labels)
        
        # Enabling grid
        ax.grid(True)

        # Return the figure object
        return fig
    
    def scatter_spacing(self, num_points_per_row):
        intercepts = np.array(self.aggregated_intercepts)

        if intercepts.size == 0:
            print("Error: No intercepts found.")
            return None, None  # Or handle this case as needed

        if len(intercepts) % num_points_per_row != 0:
            print(f"Warning: Number of intercepts ({len(intercepts)}) is not divisible by num_points_per_row ({num_points_per_row}).")
            return None, None

        num_rows = len(intercepts) // num_points_per_row
        intercepts_reshaped = intercepts.reshape((num_rows, num_points_per_row, -1))
        print(intercepts_reshaped)
        x_coords = intercepts_reshaped[:, :, 0]
        y_coords = intercepts_reshaped[:, :, 1]
        z_coords = intercepts_reshaped[:, :, 2]
    
        avg_y_differences = np.mean(np.abs(np.sqrt(np.diff(np.mean(y_coords, axis=1))**2+np.diff(np.mean(x_coords, axis=1))**2)))

        z_differences_per_row = [np.mean(np.abs(np.diff(np.sort(row)))) for row in z_coords]
        avg_z_differences = np.mean(z_differences_per_row)
        
        return avg_y_differences, avg_z_differences