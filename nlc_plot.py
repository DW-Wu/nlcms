from argparse import ArgumentParser
from glob import glob
from matplotlib.pyplot import get_cmap
import numpy as np
import os
from os.path import join, exists, basename
import sys
from vtk import *

from nlc_func import *


def biaxiality(q1, q2, q3, q4, q5):
    """tr(Q^3)^2/tr(Q^2)^3, which is between 0 and 1/6"""
    return 1 - 6 * trace_Q3(q1, q2, q3, q4, q5)**2 / (trace_Q2(q1, q2, q3, q4, q5)**3 + 1e-14)


def mkVtkCube(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """Build VTK PolyData object in the cube spanned by a mesh grid"""
    assert X.shape == Y.shape == Z.shape and len(X.shape) == 3
    cube = vtkUnstructuredGrid()
    points = vtkPoints()

    # Prepare points
    N1, N2, N3 = X.shape
    X = np.ascontiguousarray(X)
    Y = np.ascontiguousarray(Y)
    Z = np.ascontiguousarray(Z)
    x = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).transpose()

    def ijk2ind(i, j, k):
        """Translate 3-index to linear index in X,Y,Z"""
        nonlocal N2, N3
        return (i * N2 + j) * N3 + k

    voxels = [(ijk2ind(i, j, k), ijk2ind(i + 1, j, k),
               ijk2ind(i + 1, j + 1, k), ijk2ind(i, j + 1, k),
               ijk2ind(i, j, k + 1), ijk2ind(i + 1, j, k + 1),
               ijk2ind(i + 1, j + 1, k + 1), ijk2ind(i, j + 1, k + 1))
              for i in range(N1 - 1) for j in range(N2 - 1) for k in range(N3 - 1)]

    # Build grid
    for i, xi in enumerate(x):
        points.InsertPoint(i, xi)
    cube.SetPoints(points)

    for v in voxels:
        cubie = vtkHexahedron()
        for i in range(8):
            cubie.GetPointIds().SetId(i, v[i])
        cube.InsertNextCell(cubie.GetCellType(),
                            cubie.GetPointIds())
    return cube


def mkVtkCubeData(data, name):
    assert data.shape[2] == data.shape[1] == data.shape[0]
    scalars = vtkFloatArray()
    for i, vi in enumerate(data.ravel()):
        scalars.InsertTuple1(i, vi)
    scalars.SetName(name)
    return scalars


def writeVtkData(fname, data, type):
    if type == "pd":
        writer = vtkXMLPolyDataWriter()
    elif type == "ug":
        writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(fname)
    writer.SetInputData(data)
    writer.Write()


def plotBiax(X: LCState_s, N_view=None):
    if N_view is None:
        N_view = N
    xv = x.sine_trans(sz=N_view)
    biax = np.pad(biaxiality(xv.q1, xv.q2, xv.q3, xv.q4, xv.q5),
                  [(1, 1), (1, 1), (1, 1)], constant_values=0)
    phi = np.pad(xv.phi, [(1, 1), (1, 1), (1, 1)], constant_values=0)

    xx = np.arange(N_view + 2) / (N_view + 1)
    cube = mkVtkCube(*np.meshgrid(xx, xx, xx, indexing="ij"))  # prepare grid
    scalars1 = mkVtkCubeData(phi, "phi")  # prepare phi data
    cube.GetPointData().SetScalars(scalars1)
    scalars2 = mkVtkCubeData(biax, "biax")  # prepare biaxiality data
    cube.GetPointData().AddArray(scalars2)

    # Get contour surface
    contours = vtkContourGrid()
    contours.SetInputData(cube)
    contours.SetValue(0, 0.5)  # 0.5 isosurface
    contours.Update()
    surf: vtkPolyData = contours.GetOutput()  # a polydata object
    surf.GetPointData().RemoveArray(0)  # remove phi
    surf.GetPointData().RemoveArray(0)  # remove biax

    # Get interior biaxiality field
    thres = vtkThreshold()
    thres.SetInputData(cube)
    thres.SetInputArrayToProcess(1,  # index of array
                                 0,  # input port
                                 0,  # input connection
                                 1,  # field to extract (1 for CELL)
                                 "phi")  # array name
    thres.SetUpperThreshold(0.5)
    thres.SetThresholdFunction(2)  # 2 for `above upper' criterion
    thres.Update()
    biax_in: vtkUnstructuredGrid = thres.GetOutput()  # grid after threshold
    biax_in.GetPointData().RemoveArray(0)  # remove phi
    biax_in.GetPointData().SetActiveScalars("biax")

    return surf, biax_in


def fccCloud(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), width=0.01):
    """Compute point cloud from face-centered cubic packing
    Layers are parallel to the yOz plane"""
    # radius of circumsphere
    R = .5 * np.sqrt((xlim[1] - xlim[0])**2 + (ylim[1] - ylim[0])**2 + (zlim[1] - zlim[0])**2)
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


def plotDir(X, scale_factor=1., phi_thres=0.5, width=0.1, resolution=8):
    xx, yy, zz = fccCloud(width=width)  # FCC point cloud
    phi = X.values_x(xx, yy, zz, phi_only=True)
    mask = (phi > phi_thres)
    xx = xx[mask]
    yy = yy[mask]
    zz = zz[mask]
    length = len(xx)
    q1, q2, q3, q4, q5, phi = X.values_x(xx, yy, zz)

    # Information on spherical mesh
    def ij2id(i, j):
        nonlocal resolution
        return i * resolution + j

    # A list of facets on a triangulated sphere
    facets = [(ij2id(i, j), ij2id(i + 1, j), ij2id(i + 1, (j + 1) % resolution))
              for i in range(resolution) for j in range(resolution)] + \
        [(ij2id(i, j), ij2id(i + 1, (j + 1) % resolution), ij2id(i, (j + 1) % resolution))
         for i in range(1, resolution + 1) for j in range(resolution)]
    # Latitude and longitude
    ph, th = np.meshgrid(np.linspace(-.5, .5, resolution + 1) * np.pi,
                         np.arange(resolution) * 2 * np.pi / resolution, indexing="ij")
    ph = ph.ravel()
    th = th.ravel()
    sphere_x = np.cos(ph) * np.cos(th)
    sphere_y = np.cos(ph) * np.sin(th)
    sphere_z = np.sin(ph)

    poly = vtkAppendPolyData()
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
        S *= scale_factor * width * 0.5

        # Build local ellipsoidal mesh
        x_loc = (S[0, 0] * sphere_x + S[0, 1] * sphere_y + S[0, 2] * sphere_z) + xx[i]
        y_loc = (S[1, 0] * sphere_x + S[1, 1] * sphere_y + S[1, 2] * sphere_z) + yy[i]
        z_loc = (S[2, 0] * sphere_x + S[2, 1] * sphere_y + S[2, 2] * sphere_z) + zz[i]
        poly_loc = vtkPolyData()
        pt_loc = vtkPoints()
        cell_loc = vtkCellArray()
        for i in range(len(x_loc)):
            pt_loc.InsertPoint(i, (x_loc[i], y_loc[i], z_loc[i]))
        for fct in facets:
            vil = vtkIdList()
            for j in range(3):
                vil.InsertNextId(fct[j])
            cell_loc.InsertNextCell(vil)
        poly_loc.SetPoints(pt_loc)
        # Append to global mesh
        poly_loc.SetPolys(cell_loc)
        if i == 0:
            poly.SetInputData(poly_loc)
        else:
            poly.AddInputData(poly_loc)
        poly.Update()
    return poly.GetOutput()


VTK_COLORS = vtkNamedColors()


def showDroplet(surf, biax, dir, clipNormal=(1, 0, 0)):
    """Show droplet"""
    global VTK_COLORS

    # Clip plane
    plane = vtkPlane()
    plane.SetOrigin(.5, .5, .5)
    plane.SetNormal(*clipNormal)

    # clip, mapper & actor for interface
    clip1 = vtkClipPolyData()
    clip1.SetInputData(surf)
    clip1.SetClipFunction(plane)
    clip1.Update()
    mapper1 = vtkPolyDataMapper()
    mapper1.SetInputData(clip1.GetOutput())
    mapper1.ScalarVisibilityOff()
    actor1 = vtkActor()
    actor1.SetMapper(mapper1)
    actor1.GetProperty().SetRepresentationToSurface()
    actor1.GetProperty().EdgeVisibilityOff()
    actor1.GetProperty().LightingOff()
    actor1.GetProperty().SetColor(VTK_COLORS.GetColor3d('green'))

    # colormap
    lut = vtkLookupTable()
    lut.SetTableRange(0., 1.)
    # Cool to warm
    c2w = get_cmap("coolwarm")
    lut.SetNumberOfColors(256)
    for i in range(256):
        lut.SetTableValue(i, *c2w(i / 255)[0:3])
    lut.SetNanColor(1., 1., 0., 1.)  # Yellow for NAN

    # clip, mapper & actor for biax field
    clip2 = vtkClipDataSet()
    clip2.SetInputData(biax)
    clip2.SetClipFunction(plane)
    clip2.Update()
    mapper2 = vtkDataSetMapper()
    mapper2.SetInputData(clip2.GetOutput())
    mapper2.ScalarVisibilityOn()  # color map requires scalar visibility
    mapper2.SetScalarModeToUsePointData()
    mapper2.SetScalarRange(0, 1)  # scalar range for color map
    mapper2.SetColorModeToMapScalars()
    mapper2.SetLookupTable(lut)  # Use coolwarm lookup table
    actor2 = vtkActor()
    actor2.SetMapper(mapper2)
    actor2.GetProperty().SetRepresentationToSurface()
    actor2.GetProperty().EdgeVisibilityOff()
    actor2.GetProperty().LightingOff()

    # clip, mapper & actor for director field
    clip3 = vtkClipPolyData()
    clip3.SetInputData(dir)
    clip3.SetClipFunction(plane)
    clip3.SetValue(-0.06)
    clip3.Update()
    mapper3 = vtkPolyDataMapper()
    mapper3.SetInputData(clip3.GetOutput())
    mapper3.ScalarVisibilityOff()
    actor3 = vtkActor()
    actor3.SetMapper(mapper3)
    actor3.GetProperty().SetRepresentationToSurface()
    actor3.GetProperty().EdgeVisibilityOff()
    # actor3.GetProperty().LightingOff()
    actor3.GetProperty().SetDiffuse(0.7)
    actor3.GetProperty().SetSpecular(0.4)
    actor3.GetProperty().SetSpecularPower(20)
    actor3.GetProperty().SetColor(VTK_COLORS.GetColor3d('white'))

    # A renderer and render window
    renderer = vtkRenderer()
    renderer.SetBackground(VTK_COLORS.GetColor3d("transparent"))

    # add the actors
    renderer.AddActor(actor1)
    renderer.AddActor(actor2)
    renderer.AddActor(actor3)
    renderer.SetLightFollowCamera(1)

    # set camera
    cam = renderer.GetActiveCamera()
    cam.SetPosition(-1., 0.5, 0.5)
    cam.SetFocalPoint(.5, .5, .5)  # aim at center of droplet
    cam.SetViewUp(0., 0., 1.)  # correct orientation

    # render window
    renwin = vtkRenderWindow()
    renwin.AddRenderer(renderer)
    renwin.SetWindowName('NLC Droplet')
    renwin.SetSize(1000, 1000)

    return renwin


parser = ArgumentParser(prog="nlc_plot",
                        description="Plot 3D NLC state using VTK and absolutely no Mayavi")
parser.add_argument("files", nargs='*', action="store", help="Input file(s)")
parser.add_argument("-N", "--num_view", action="store", default=95,
                    help="Grid size of final view")
parser.add_argument("-o", "--output", action="store", default="out",
                    help="Output folder name")
parser.add_argument("--phi-thres", action="store", default=0.5, type=float,
                    help="Threshold value of phi")
parser.add_argument("--no-biax", action="store_true", default=False,
                    help="Do not plot biaxiality")
parser.add_argument("--no-dir", action="store_true", default=False,
                    help="Do not plot directors")
parser.add_argument("--show", '-S', action="store_true", default=False,
                    help="Show with internal subroutine")


if __name__ == "__main__":
    args = parser.parse_args()

    OUTD = join(os.path.abspath('.'), args.output)
    if not exists(OUTD):
        os.mkdir(OUTD)

    # Gather input files
    FL = []
    for a in args.files:
        FL += glob(a)
    if len(FL) > 1:
        args.show = False  # Do not show multiple files
    for fn in FL:
        if not fn.endswith('.npy'):
            raise ValueError("Invalid state file name")
        x = load_lc(fn)
        # Plot biaxiality
        if not args.no_biax:
            surf, biax = plotBiax(x, int(args.num_view))  # phi=0.5 isosurface and interior biaxiality
            writeVtkData(join(OUTD, "interf_%s.vtp" % basename(fn).removesuffix('.npy')),
                         surf, type="pd")
            writeVtkData(join(OUTD, "biax_%s.vtu" % basename(fn).removesuffix('.npy')),
                         biax, type="ug")
        if not args.no_dir:
            poly = plotDir(x, scale_factor=0.5, phi_thres=args.phi_thres, width=0.1)
            writeVtkData(join(OUTD, "dir_%s.vtp" % basename(fn).removesuffix('.npy')),
                         poly, type="pd")
        if not args.no_biax and not args.no_dir:
            win = showDroplet(surf, biax, poly, clipNormal=(1, 0, 0))
            if args.show:
                # An interactor
                interactor = vtkRenderWindowInteractor()
                interactor.SetRenderWindow(win)
                # Start
                interactor.Initialize()
                win.Render()
                interactor.Start()
            # Save scene to image
            win.Render()
            win.SetAlphaBitPlanes(1)  # enable alpha channel (transparency)
            img = vtkWindowToImageFilter()
            img.SetInput(win)
            # img.SetScale(1, 1)
            img.SetInputBufferTypeToRGBA()
            img.ReadFrontBufferOff()
            img.Update()
            writer = vtkPNGWriter()
            writer.SetFileName(join(OUTD, "scene_%s.png" % basename(fn).removesuffix('.npy')))
            writer.SetInputConnection(img.GetOutputPort())
            writer.Write()
