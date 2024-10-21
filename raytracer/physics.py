"""The module for all the physics formulas related to ray tracing"""

import numpy as np
from raytracer.utils import norm


def refract(direc, normal, n_1, n_2):
    """
    Calcalate the direction of a refracted ray using Snell's law

    Args:
        direc (list or numpy.ndarray): A 3-element list or array representing direction vector of the incident ray.
        normal (list or numpy.ndarray): A 3 element list or array representing normal vector of surface at the point
            of incidence.
        n_1 (float): The refractive index of the medium before incidence.
        n_2 (float): The retractive index of the medium after incidence.

    Returns:
        numpy.ndarray: A normalized 3-element array representing the direction vector of the refracted ray if refraction
            occurs successfully.
        None: if total internal reflection occurs, which happens when the angle of incidence exceeds the critical angle.
    """
    direc = norm(direc)
    normal = norm(normal)

    cos_theta1 = np.inner(direc, normal)
    if cos_theta1 > 0:
        normal = - normal
    cos_theta1 = abs(cos_theta1)

    sin_theta1 = np.sqrt(1 - cos_theta1 ** 2)

    if sin_theta1 > n_2 / n_1:
        return None

    sin_theta2 = n_1 * sin_theta1 / n_2
    cos_theta2 = np.sqrt(1 - sin_theta2 ** 2)

    refrac = (n_1 / n_2) * direc + ((n_1 / n_2)
                                    * cos_theta1 - cos_theta2) * normal
    refrac = norm(refrac)
    return refrac


def reflect(direc, normal):
    """
    Calcalate the direction of a reflected ray using linear algebra

    Args:
        direc (list or numpy.ndarray): A 3-element list or array representing direction vector of the incident ray.
        normal (list or numpy.ndarray): A 3 element list or array representing normal vector of surface at the point
            of incidence.

    Returns:
        numpy.ndarray: A normalized 3-element array representing the direction vector of the reflected ray
    """
    direc = norm(direc)
    normal = norm(normal)

    i_dot_n = np.dot(direc, normal)
    if i_dot_n < 0:
        normal = - normal
    i_dot_n = abs(i_dot_n)

    reflec = direc - 2 * i_dot_n * normal
    reflec = norm(reflec)
    return reflec


def reflectivity(direc, normal, n_1, n_2):
    """
    Calculate reflectivity using Fresnel's equation.

    Args:
        direc (list or numpy.ndarray): A 3-element list or array representing direction vector of the incident ray.
        normal (list or numpy.ndarray): A 3 element list or array representing normal vector of surface at the point
            of incidence.
        n_1 (float): The refractive index of the medium before incidence.
        n_2 (float): The retractive index of the medium after incidence.

    Returns:
        float: the value of Fresnel's reflectivity
    """
    direc = norm(direc)
    normal = norm(normal)

    cos_theta1 = abs(np.inner(direc, normal))
    sin_theta1 = np.sqrt(1 - cos_theta1 ** 2)

    if sin_theta1 > n_2 / n_1:
        return 1.0  # total internal reflection case

    sin_theta2 = n_1 * sin_theta1 / n_2
    cos_theta2 = np.sqrt(1 - sin_theta2 ** 2)

    R_s = ((n_1 * cos_theta1 - n_2 * cos_theta2) /
           (n_1 * cos_theta1 + n_2 * cos_theta2))**2

    R_p = ((n_1 * cos_theta2 - n_2 * cos_theta1) /
           (n_1 * cos_theta2 + n_2 * cos_theta1))**2

    return (R_s + R_p) / 2


def smellmeier_equation(lamda):
    """
    Caculate the refractive index of borosilicate crown glass using smellmeier equation.

    Args:
        lamda (float): The wavelength of incident ray

    Returns:
        float: The refractive index depending on the wavelength
    """
    B1 = 1.03961212
    B2 = 0.231792344
    B3 = 1.01046945
    C1 = 6.00069867E-3
    C2 = 2.00179144E-2
    C3 = 1.03560653E2

    n_sq = 1 + B1 * lamda ** 2 / (lamda ** 2 - C1) + \
        B2 * lamda ** 2 / (lamda ** 2 - C2) + \
        B3 * lamda ** 2 / (lamda ** 2 - C3)

    return np.sqrt(n_sq)
