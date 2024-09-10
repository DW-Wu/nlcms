import sys, os
import scipy
from os.path import join
from argparse import ArgumentParser

# Parse argument first
# Do not load mayavi engine if only asking for help message
if __name__ == "__main__":
    parser = ArgumentParser(prog="nlc_plot",
                            description="Plot 3D NLC state using Mayavi engine")
    parser.add_argument("file", action="store", help="Input file")
    parser.add_argument("-N", "--num_view", action="store", default=63,
                        help="Grid size of final view")
    parser.add_argument("-o", "--output", action="store", default="out",
                        help="Output folder name")
    parser.add_argument("--no-biax", action="store_true", default=False,
                        help="Do not plot biaxiality (which is very large)")
    parser.add_argument("--no-dir", action="store_true", default=False,
                        help="Do not plot directors")
    parser.add_argument("--no-phi", action="store_true", default=False,
                        help="Do not plot φ contours")
    args = parser.parse_args()

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mayavi import mlab
from tvtk.api import tvtk, write_data

from nlc_state import *
from nlc_func import trace_Q2, trace_Q3


def points_fcc(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), width=0.01):
    """Compute point cloud from face-centered cubic packing
    Layers are parallel to the yOz plane"""
    # radius of circumsphere
    R = .5 * np.sqrt((xlim[1] - xlim[0]) ** 2 + (ylim[1] - ylim[0]) ** 2 + (zlim[1] - zlim[0]) ** 2)
    N_cubes = int(np.sqrt(2) * R / width + 1)  # number of small cubes
    a = width * np.sqrt(2)  # side length of cube cell
    x = np.arange(N_cubes)
    xx, yy, zz = np.meshgrid(x, x, x)
    # orthogonal frame
    o1 = a * np.array([np.sqrt(1. / 3), np.sqrt(2. / 3), 0])
    o2 = a * np.array([np.sqrt(1. / 3), -np.sqrt(1. / 6), np.sqrt(1. / 2)])
    o3 = a * np.array([np.sqrt(1. / 3), -np.sqrt(1. / 6), -np.sqrt(1. / 2)])
    # vertex of grid
    x0 = (xlim[0] + xlim[1]) / 2 - np.sqrt(3) * R
    y0 = (ylim[0] + ylim[1]) / 2
    z0 = (zlim[0] + zlim[1]) / 2
    # form grids (4 in total)
    # the grid is large enough to contain the desired range
    xx_1 = x0 + xx * o1[0] + yy * o2[0] + zz * o3[0]
    yy_1 = y0 + xx * o1[1] + yy * o2[1] + zz * o3[1]
    zz_1 = z0 + xx * o1[2] + yy * o2[2] + zz * o3[2]
    xx_2 = xx_1 + .5 * (o1[0] + o2[0])
    yy_2 = yy_1 + .5 * (o1[1] + o2[1])
    zz_2 = zz_1 + .5 * (o1[2] + o2[2])
    xx_3 = xx_1 + .5 * (o1[0] + o3[0])
    yy_3 = yy_1 + .5 * (o1[1] + o3[1])
    zz_3 = zz_1 + .5 * (o1[2] + o3[2])
    xx_4 = xx_1 + .5 * (o2[0] + o3[0])
    yy_4 = yy_1 + .5 * (o2[1] + o3[1])
    zz_4 = zz_1 + .5 * (o2[2] + o3[2])
    X = np.hstack([xx_1.ravel(), xx_2.ravel(), xx_3.ravel(), xx_4.ravel()])
    Y = np.hstack([yy_1.ravel(), yy_2.ravel(), yy_3.ravel(), yy_4.ravel()])
    Z = np.hstack([zz_1.ravel(), zz_2.ravel(), zz_3.ravel(), zz_4.ravel()])
    R = (xlim[0] < X) & (X < xlim[1]) & (ylim[0] < Y) & (Y < ylim[1]) \
        & (zlim[0] < Z) & (Z < zlim[1])
    return X[R], Y[R], Z[R]


def plot_phi(X: LCState_s, figure=None, levels=5, sz=None):
    if figure is None:
        figure = mlab.gcf()
    if sz is None:
        sz = X.N
    xxx = np.arange(1, sz + 1) / (sz + 1)
    phi = X.phi_values(sz=sz)
    xx, yy, zz = np.meshgrid(xxx, xxx, xxx, indexing='ij')
    surf = mlab.contour3d(xx, yy, zz, phi, figure=figure, contours=levels,
                          color=(.7, .7, .7), opacity=.5)
    return surf


def plot_biax(X: LCState_s, figure=None, N_view=None, phi_thres=0.5, color_resolution=8):
    """Plot biaxiality order parameter as scalar field in space
    `N_view`: number of grid points along one edge
    `phi_thres`: threshold value of phi (above which grid points are visible)
    `color_resolution`: length of the (discrete) color scale"""
    if figure is None:
        figure = mlab.gcf()
    if N_view is None:
        N_view = X.N
    xxx = np.arange(1, N_view + 1) / (1 + N_view)
    xx, yy, zz = np.meshgrid(xxx, xxx, xxx, indexing='ij')
    X_v = X.sine_trans(sz=N_view)
    trQ2 = trace_Q2(X_v.q1, X_v.q2, X_v.q3, X_v.q4, X_v.q5)
    trQ3 = trace_Q3(X_v.q1, X_v.q2, X_v.q3, X_v.q4, X_v.q5)
    biax = 1 - 6 * trQ3 ** 2 / (trQ2 ** 3 + 1e-14)
    mask = (X_v.phi > phi_thres)
    if not np.any(mask):
        raise Warning("No φ")
    xx = xx[mask]
    yy = yy[mask]
    zz = zz[mask]
    biax = biax[mask]
    # b_steps = np.linspace(np.min(biax), np.max(biax), color_resolution + 1)
    # c = cm.RdBu(Normalize()(b_steps))
    kwargs = dict(extent=[0, 1, 0, 1, 0, 1], figure=figure,
                  mode='cube', colormap="RdBu", opacity=0.1, reset_zoom=False,
                  scale_factor=1. / (N_view + 1), scale_mode='none')
    img = mlab.points3d(xx, yy, zz, biax, **kwargs)
    return tvtk.to_tvtk(img.actor.actors[0].mapper.input)


def plot_director(X: LCState_s, figure=None,
                  scale_factor=1., phi_thres=0.5, density=0.05, resolution=8):
    """Visualization of solution by Hu (2016).
    Contours of biaxiality order parameter
    Q tensor represented by ellipsoids"""
    if figure is None:
        figure = mlab.gcf()
    w = 1. / X.N / density ** (1 / 3)
    xx, yy, zz = points_fcc(width=w)  # FCC point cloud
    phi = X.values_x(xx, yy, zz, phi_only=True)
    # restrict to phi>threshold
    mask = (phi > phi_thres)
    xx = xx[mask]
    yy = yy[mask]
    zz = zz[mask]
    length = len(xx)
    q1, q2, q3, q4, q5, phi = X.values_x(xx, yy, zz)
    trQ2 = trace_Q2(q1, q2, q3, q4, q5)
    trQ3 = trace_Q3(q1, q2, q3, q4, q5)
    biax = 1 - 6 * trQ3 ** 2 / (trQ2 ** 3 + 1e-14)
    c = cm.viridis(Normalize()(biax))
    # V=np.zeros([length,3]) # storage for eigenvectors
    # unit sphere coordinates
    ph, th = np.meshgrid(np.linspace(-.5, .5, resolution + 1) * np.pi,
                         np.linspace(0, 2, resolution + 1) * np.pi)
    sphere_x = np.cos(ph) * np.cos(th)
    sphere_y = np.cos(ph) * np.sin(th)
    sphere_z = np.sin(ph)

    append_pd = tvtk.AppendPolyData()  # tvtk object that merges multiple datasets
    for i in range(length):
        # Get Q tensor at point
        Q = np.array([[q1[i] - q3[i], q2[i], q4[i]],
                      [q2[i], -q1[i] - q3[i], q5[i]],
                      [q4[i], q5[i], 2 * q3[i]]], dtype="float64")
        lam, S = np.linalg.eig(Q)
        # normalize eigenvalues (use positive value to make ellipsoid visible)
        lam = (lam - np.min(lam)) / (np.max(lam) - np.min(lam) + 1e-10) + .1
        for k in range(3):
            S[:, k] *= lam[k]
        S *= scale_factor * w * 0.5
        surf = mlab.mesh((S[0, 0] * sphere_x + S[0, 1] * sphere_y + S[0, 2] * sphere_z) + xx[i],
                         (S[1, 0] * sphere_x + S[1, 1] * sphere_y + S[1, 2] * sphere_z) + yy[i],
                         (S[2, 0] * sphere_x + S[2, 1] * sphere_y + S[2, 2] * sphere_z) + zz[i],
                         color=tuple(c[i, 0:3]), scalars=None,
                         figure=figure)
        localpoly = tvtk.to_tvtk(surf.actor.actors[0].mapper.input)  # get polydata
        scalar_array = np.ones(localpoly.number_of_cells) * biax[i]
        localpoly.cell_data.reset()
        localpoly.cell_data.scalars = scalar_array.ravel()
        localpoly.cell_data.update()  # write scalar to cells
        if i == 0:
            append_pd.set_input_data(localpoly)  # first input data set
        else:
            append_pd.add_input_data(localpoly)  # trailing input data sets
        append_pd.update()  # always update
    return append_pd.output


if __name__ == "__main__":
    # # test FCC point cloud
    # X, Y, Z = points_fcc(width=0.1)
    # mlab.points3d(X, Y, Z, scale_factor=0.1)
    # mlab.show()
    args = parser.parse_args()

    OUTD = join(os.path.dirname(sys.argv[0]), args.output)
    if not os.path.exists(OUTD):
        os.mkdir(OUTD)

    # Prepare state
    X = load_lc(args.file)

    # Plot
    s = int(args.num_view)
    X_v = X.sine_trans(sz=s)
    fig = mlab.figure(1)
    xxx = np.arange(1, s + 1) / (s + 1)
    xx, yy, zz = np.meshgrid(xxx, xxx, xxx, indexing='ij')
    if not args.no_phi:
        plot_phi(X, figure=fig)
        mlab.savefig(join(OUTD, "phi.wrl"), figure=fig)

    mlab.clf(fig)
    if not args.no_dir:
        dir_vtk = plot_director(X, figure=fig, scale_factor=0.5, phi_thres=.6, density=0.05)
        write_data(dir_vtk, join(OUTD, "dir.vtp"))
        # mlab.savefig(join(OUTD, "dir.obj"), figure=fig)

    mlab.clf(fig)
    if not args.no_biax:
        biax_vtk = plot_biax(X, figure=fig, N_view=127, phi_thres=.8, color_resolution=16)
        write_data(biax_vtk, join(OUTD, "biax.vtp"))
