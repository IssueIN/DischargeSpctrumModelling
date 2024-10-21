"""Analysis module."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from raytracer.elements import SphericalRefraction, OutputPlane, SphericalSurface
from raytracer.rays import Ray, RayBundle, RaySeries
from raytracer.lenses import PlanoConvex, ConvexPlano
from raytracer.physics import smellmeier_equation
from raytracer.utils import plot_ray_series, savefig, piecewise_linear_fit, quad_fit


def task8():
    """
    Task 8.

    In this function you should check your propagate_ray function properly
    finds the correct intercept and correctly refracts a ray. Don't forget
    to check that the correct values are appended to your Ray object.
    """
    ray = Ray(pos=[0., 0., 0.], direc=[0., 0., 1.])
    ray2 = Ray(pos=[0., 0., 1.], direc=[0., 0., 1.])
    sr = SphericalRefraction(
        z_0=5.,
        aperture=3.,
        curvature=2.,
        n_1=1.5,
        n_2=1.)
    sr.propagate_ray(ray)
    sr.propagate_ray(ray2)


def task10():
    """
    Task 10.

    In this function you should create Ray objects with the given initial positions.
    These rays should be propagated through the surface, up to the output plane.
    You should then plot the tracks of these rays.
    This function should return the matplotlib figure of the ray paths.

    Returns:
        Figure: the ray path plot.
    """
    sr = SphericalRefraction(100, 100, 0.03, 1.0, 1.5)
    opp = OutputPlane(250)
    rays = [
        Ray([0, 4, 0], [0, 0, 1]),
        Ray([0, 1, 0], [0, 0, 1]),
        Ray([0, 0.2, 0], [0, 0, 1]),
        Ray([0, 0, 0], [0, 0, 1]),
        Ray([0, -0.2, 0], [0, 0, 1]),
        Ray([0, -1, 0], [0, 0, 1]),
        Ray([0, -4, 0], [0, 0, 1])
    ]
    for ray in rays:
        sr.propagate_ray(ray)
        opp.propagate_ray(ray)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for ray in rays:
        verts = np.array(ray.vertices())
        ax.plot3D(verts[:, 0], verts[:, 1], verts[:, 2], label='Ray Path')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')

    return fig


def task11():
    """
    Task 11.

    In this function you should propagate the three given paraxial rays through the system
    to the output plane and the tracks of these rays should then be plotted.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for ray paths
    2. the calculated focal point.

    Returns:
        tuple[Figure, float]: the ray path plot and the focal point
    """
    rays = [
        Ray([0.1, 0.1, 0], [0, 0, 1]),
        Ray([0, 0, 0], [0, 0, 1]),
        Ray([-0.1, -0.1, 0], [0, 0, 1])
    ]
    sr = SphericalRefraction(100, 100, 0.03, 1.0, 1.5)
    opp = OutputPlane(sr.focal_point())

    f = []

    for ray in rays:
        sr.propagate_ray(ray)
        if ray.direc()[0] != 0:
            x_diff = 0 - ray.pos()[0]
            l = x_diff / ray.direc()[0]
            f_z = ray.pos()[2] + l * ray.direc()[2]
            f.append(f_z)

        opp.propagate_ray(ray)

    f_cal = sum(f) / len(f)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')

    for ray in rays:
        verts = np.array(ray.vertices())
        ax.plot3D(verts[:, 0], verts[:, 1], verts[:, 2])

    # assert np.isclose(f_cal, sr.focal_point())

    return fig, f_cal


def task12():
    """
    Task 12.

    In this function you should create a RayBunble and propagate it to the output plane
    before plotting the tracks of the rays.
    This function should return the matplotlib figure of the track plot.

    Returns:
        Figure: the track plot.
    """
    sr = SphericalRefraction(100, 100, 0.03, 1.0, 1.5)
    opp = OutputPlane(sr.focal_point())
    ray_bundle = RayBundle()

    ray_bundle.propagate_bundle([sr, opp])
    fig = ray_bundle.track_plot()

    return fig


def task13():
    """
    Task 13.

    In this function you should again create and propagate a RayBundle to the output plane
    before plotting the spot plot.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the spot plot
    2. the simulation RMS

    Returns:
        tuple[Figure, float]: the spot plot and rms
    """
    sr = SphericalRefraction(100, 100, 0.03, 1.0, 1.5)
    opp = OutputPlane(sr.focal_point())
    ray_bundle = RayBundle()

    ray_bundle.propagate_bundle([sr, opp])
    fig = ray_bundle.spot_plot()
    rms = ray_bundle.rms()

    return fig, rms


def task14():
    """
    Task 14.

    In this function you will trace a number of RayBundles through the optical system and
    plot the RMS and diffraction scale dependence on input beam radii.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the diffraction scale plot
    2. the simulation RMS for input beam radius 2.5
    3. the diffraction scale for input beam radius 2.5

    Returns:
        tuple[Figure, float, float]: the plot, the simulation RMS value, the diffraction scale.
    """
    sr = SphericalRefraction(100, 100, 0.03, 1.0, 1.5)
    opp = OutputPlane(sr.focal_point())
    wl = 588E-6
    fl = (sr.focal_point() - sr.z_0())

    r_list = np.linspace(0.1, 10, 100)
    dx = [wl * fl / (2 * r) for r in r_list]
    ray_bundles = [RayBundle(rmax=r) for r in r_list]

    rms_values = []
    for rb in ray_bundles:
        rb.propagate_bundle([sr, opp])
        rms_values.append(rb.rms())

    index = np.searchsorted(r_list, 2.5)
    dx_2_5 = dx[index]
    rms_2_5 = rms_values[index]

    fig, ax1 = plt.subplots()
    plt.grid()

    ax1.set_xlabel('Radius(mm)')
    ax1.set_ylabel('RMS (mm)')
    ax1.scatter(r_list, rms_values, label='RMS', color='orange', s=9)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Diffraction Scale (mm)')
    ax2.scatter(r_list, dx, label='Diffraction Scale', s=9)

    ymin = min(*rms_values, *dx) - 0.01
    ymax = max(*rms_values, *dx) + 0.01
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    return fig, rms_2_5, dx_2_5


def task15():
    """
    Task 15.

    In this function you will create plano-convex lenses in each orientation and propagate a RayBundle
    through each to their respective focal point. You should then plot the spot plot for each orientation.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the spot plot for the plano-convex system
    2. the focal point for the plano-convex lens
    3. the matplotlib figure object for the spot plot for the convex-plano system
    4  the focal point for the convex-plano lens


    Returns:
        tuple[Figure, float, Figure, float]: the spot plots and rms for plano-convex and convex-plano.
    """
    # wl = 588E-6
    pc = PlanoConvex(100., -0.02, 1.5168, 1., 5., 50.)
    cp = ConvexPlano(100., 0.02, 1.5168, 1., 5., 50.)
    ray_bundle_pc = RayBundle(rmax=5)
    ray_bundle_cp = RayBundle(rmax=5)
    opp_pc = OutputPlane(pc.focal_point())
    opp_cp = OutputPlane(cp.focal_point())

    ray_bundle_pc.propagate_bundle([pc, opp_pc])
    ray_bundle_cp.propagate_bundle([cp, opp_cp])

    fig_pc = ray_bundle_pc.spot_plot()
    fig_cp = ray_bundle_cp.spot_plot()

    f_pc = pc.focal_point()
    f_cp = cp.focal_point()
    return fig_pc, f_pc, fig_cp, f_cp


def task16():
    """
    Task 16.

    In this function you will be again plotting the radial dependence of the RMS and diffraction values
    for each orientation of your lens.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the diffraction scale plot
    2. the RMS for input beam radius 3.5 for the plano-convex system
    3. the RMS for input beam radius 3.5 for the convex-plano system
    4  the diffraction scale for input beam radius 3.5

    Returns:
        tuple[Figure, float, float, float]: the plot, RMS for plano-convex, RMS for convex-plano, diffraction scale.
    """
    wl = 588E-6
    pc = PlanoConvex(100., -0.02, 1.5168, 1., 5., 50.)
    cp = ConvexPlano(100., 0.02, 1.5168, 1., 5., 50.)
    opp_pc = OutputPlane(pc.focal_point())
    opp_cp = OutputPlane(cp.focal_point())
    print(pc.focal_point())

    # Effective Focal Length, which is equal for both planoconvex and
    # convexplano lens
    efl = (pc.focal_point() - pc.z_0()) - pc.thickness()

    r_list = np.linspace(0.1, 10, 100)
    dx = [wl * efl / (2 * r) for r in r_list]

    ray_bundles_pc = [RayBundle(rmax=r) for r in r_list]
    ray_bundles_cp = [RayBundle(rmax=r) for r in r_list]

    rms_values_pc = []
    for rb in ray_bundles_pc:
        rb.propagate_bundle([pc, opp_pc])
        rms_values_pc.append(rb.rms())

    rms_values_cp = []
    for rb in ray_bundles_cp:
        rb.propagate_bundle([cp, opp_cp])
        rms_values_cp.append(rb.rms())

    index = np.searchsorted(r_list, 3.5)
    dx_3_5 = dx[index]
    rms_3_5_pc = rms_values_pc[index]
    rms_3_5_cp = rms_values_cp[index]

    fig, ax1 = plt.subplots()
    plt.grid()

    ax1.set_xlabel('Radius(mm)')
    ax1.set_ylabel('RMS (mm)')
    ax1.scatter(
        r_list,
        rms_values_pc,
        label='RMS for PlanoConvex lens',
        c='blue', s=9)
    ax1.scatter(
        r_list,
        rms_values_cp,
        label='RMS for ConvexPlano lens',
        c='orange', s=9)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Diffraction Scale (mm)')
    ax2.scatter(
        r_list,
        dx,
        label='Diffraction Scale',
        c='green', s=9)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0.55, 1))
    return fig, rms_3_5_pc, rms_3_5_cp, dx_3_5


def RaySeriesTest():
    """
        Test the Ray seris propagation.

        Returns:
            Figure: The figure consisting the track plot of the ray.
    """
    sr = SphericalSurface(100, 100, 0.03, 1.0, 1.5)
    opp = OutputPlane(250)
    opp2 = OutputPlane(-10)
    rays = [
        RaySeries(pos=[0, 4, 0], direc=[0, 0, 1]),
        RaySeries(pos=[0, 1, 0], direc=[0, 0, 1]),
        RaySeries(pos=[0, 0.2, 0], direc=[0, 0, 1]),
        RaySeries(pos=[0, 0, 0], direc=[0, 0, 1]),
        RaySeries(pos=[0, -0.2, 0], direc=[0, 0, 1]),
        RaySeries(pos=[0, -1, 0], direc=[0, 0, 1]),
        RaySeries(pos=[0, -4, 0], direc=[0, 0, 1])
    ]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for ray in rays:
        sr.propagate_raySeries(ray)
        opp.propagate_raySeries(ray)
        opp2.propagate_raySeries(ray)

        plot_ray_series(ax, ray)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Ray Paths')
    plt.legend()
    return fig


def rms_track(lens=ConvexPlano(100., 0.02, 1.5168, 1., 5., 50.)):
    """
    track the rms values along the z axis.

    Args:
        lens (Lens): The lens object that the ray passing through.

    Returns:
        Figure: A plot shows rms values against z-coordinates.
    """
    fl = lens.focal_point()
    z_list = np.linspace(fl - 5, fl + 5, 100)
    opp_list = [OutputPlane(z) for z in z_list]

    r_list = np.linspace(0.1, 10, 100)

    ray_bundles = [RayBundle(rmax=r) for r in r_list]

    rms_values = []
    for rb in ray_bundles:
        rb.propagate_bundle([lens])

        rms = []
        for opp in opp_list:
            rb.propagate_bundle([opp])
            rms.append(rb.rms())
        rms_values.append(rms)

    fig, ax = plt.subplots()

    ax.set_xlabel('z-coordinates(mm)')
    ax.set_ylabel('RMS (mm)')
    # for i, r in enumerate(r_list):
    #     ax1.scatter(z_list, rms_values[i], label=f'RMS for R  = {r}mm')
    ax.scatter(z_list, rms_values[1])

    return fig


def finding_focal_length(wl):
    """
    The function for finding the focal length for differenct wavelength

    Args:
        wl (float): teh wavelength of the incident ray.

    Returns:
        list: The values of fitted peak_x and its covariance.
    """
    # The smellmeier equation required the wavelength in the unit of μm
    n_lamda = smellmeier_equation(wl)
    pc = PlanoConvex(100., -0.02, n_lamda, 1., 5., 50.)

    fl = pc.focal_point()
    z_list = np.linspace(fl - 5, fl + 5, 200)
    opp_list = [OutputPlane(z) for z in z_list]
    rb = RayBundle()

    rb.propagate_bundle([pc])

    rms = []
    for opp in opp_list:
        rb.propagate_bundle([opp])
        rms.append(rb.rms())

    popt, pcov = curve_fit(
        piecewise_linear_fit, z_list, rms, p0=(
            0.4, 201, 0.02))

    return [popt[1], pcov[1][1]]


def dispersion_relation():
    """
        The function for plotting dispersion relation.
    """
    wls = np.linspace(360E-3, 830E-3, 50)

    ns = [smellmeier_equation(wl) for wl in wls]
    pcs = [PlanoConvex(100., -0.02, n, 1., 5., 50.) for n in ns]
    fl_theory = [pc.focal_point() for pc in pcs]

    parameters = [finding_focal_length(wl) for wl in wls]

    fl_simu = []
    fl_simu_cov = []
    for parameter in parameters:
        fl_simu.append(parameter[0])
        fl_simu_cov.append(parameter[1])

    plt.errorbar(
        wls,
        fl_simu,
        yerr=np.sqrt(fl_simu_cov),
        label='simulated focal point',
        fmt='o',
        markersize=3.5,
        capsize=3)
    plt.scatter(
        wls,
        fl_theory,
        label='theoretical focal point',
        s=9,
        c='orange')
    plt.grid()
    plt.xlabel('wavelength (μm)')
    plt.ylabel('focal point (mm)')
    plt.legend()
    plt.show()


def colors_track_plot():
    """
    The track plot associated with varies colors.
    """
    wls = [400E-3, 425E-3, 470E-3, 550E-3, 600E-3, 630E-3, 665E-3]
    ns = [smellmeier_equation(wl) for wl in wls]
    pcs = [PlanoConvex(100., -0.02, n, 1., 5., 50.) for n in ns]
    opp = OutputPlane(205.)
    cs = ['violet', 'Indigo', 'blue', 'green', 'yellow', 'orange', 'red']

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for i, pc in enumerate(pcs):
        rb = RayBundle()
        rb.propagate_bundle([pc, opp])
        for ray in rb.ray_bundle():
            verts = np.array(ray.vertices())
            ax.plot3D(verts[:, 0], verts[:, 1], verts[:, 2], c=f'{cs[i]}')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    plt.show()


if __name__ == "__main__":

    # # Run task 8 function
    # task8()

    # # Run task 10 function
    # FIG10 = task10()
    # savefig(FIG10, 'T-10')

    # # Run task 11 function
    # FIG11, FOCAL_POINT = task11()
    # savefig(FIG11, 'T-11')

    # # Run task 12 function
    # FIG12 = task12()
    # savefig(FIG12, 'T-12')

    # # Run task 13 function
    # FIG13, TASK13_RMS = task13()
    # savefig(FIG13, 'T-13')

    # # Run task 14 function
    # FIG14, TASK14_RMS, TASK14_DIFF_SCALE = task14()
    # savefig(FIG14, 'T-14')

    # # Run task 15 function
    # FIG15_PC, FOCAL_POINT_PC, FIG15_CP, FOCAL_POINT_CP = task15()
    # savefig(FIG15_PC, 'T-15-PC')
    # savefig(FIG15_CP, 'T-15-CP')

    # # Run task 16 function
    # FIG16, PC_RMS, CP_RMS, TASK16_DIFF_SCALE = task16()
    # savefig(FIG16, 'T-16')

    plt.show()
