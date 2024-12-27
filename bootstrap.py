"""Bootstraping the stable states from simple initial values"""

from nlc_func import *
from nlc_solve import LCSolve
import datetime
from os.path import exists
from argparse import ArgumentParser

parser = ArgumentParser(prog="bootstrap",
                        description="Bootstrap construction of special solutions")
parser.add_argument("name", choices=["radial", "ring", "score", "tactoid",
                                     "pointy", "twodrops"],
                    help="Name of solution")
parser.add_argument("-s", "--silent", action="store_true", default=False,
                    help="Silent run (no progress bars)")

c = load_lc_config("ring.json")
c = LCConfig(A=-1500, lam=2e-7, v0=0.1,
             eps=0.005, omega=20, wp=1, wv=0.5)  # same
F = LCFunc_s(**c)
N = 47
xx = np.arange(1, N + 1) / (N + 1)
xx, yy, zz = np.meshgrid(xx, xx, xx, indexing='ij')
args = parser.parse_args()

if args.name == "ring":
    # Testified bootstrap of ring (volume=0.1)
    # start from an oblate sphere
    r = np.sqrt((xx - .5)**2 + (yy - .5)**2 + 1.1 * (zz - .5)**2)
    phiv = (np.tanh((0.287941 - r) / 0.04) + 1)
    X0 = LCState_s(N)
    X0.phi[:] = 4. * fft.idstn(phiv, type=1)
    solver = LCSolve(outdir='.', conf=c, N=N, x0=X0, load_file=False,
                     verbose=not args.silent)
    print("search for ring starts", datetime.datetime.now())
    solver.solve(method='gd', metric="h1", maxiter=10000, eta=1e-4, tol=1e-8,
                 bb=True, verbose=not args.silent)  # H1 grad descent
    solver.solve(method='gd', metric="l2", maxiter=10000, eta=1e-4, tol=1e-8,
                 bb=True, verbose=not args.silent)  # L2 grad descent (different)
    solver.snapshot(fname="ring%d" % (N + 1))
    print("ring found", datetime.datetime.now())

if args.name == "score":
    # Testified bootstrap of split-core (volume=0.1)
    # Start from a prolate sphere
    r = np.sqrt((xx - .5)**2 + (yy - .5)**2 + 0.8 * (zz - .5)**2)
    phiv = (np.tanh((0.287941 - r) / 0.04) + 1)
    X0 = LCState_s(N)
    X0.phi[:] = 4. * fft.idstn(phiv, type=1)
    solver = LCSolve(outdir='.', conf=c, N=N, x0=X0, load_file=False,
                     verbose=not args.silent)
    print("search for split-core starts", datetime.datetime.now())
    # Stabilize Q
    solver.solve(method='Q-gd', maxiter=600, eta=1e-4, tol=1e-8, bb=True,
                 verbose=not args.silent)
    solver.solve(method='gd', metric="h1", maxiter=10000, eta=5e-4, tol=1e-8,
                 bb=True, verbose=not args.silent)
    solver.solve(method='gd', metric="l2", maxiter=10000, eta=5e-4, tol=1e-8,
                 bb=True, verbose=not args.silent)
    solver.snapshot(fname="score%d" % (N + 1))
    print("split-core found", datetime.datetime.now())

if args.name == "radial":
    # Testified bootstrap of radial solution (volume=0.1)
    r = np.sqrt((xx - .5)**2 + (yy - .5)**2 + (zz - .5)**2)
    phiv = (np.tanh((0.287941 - r) / 0.01) + 1)
    X0 = LCState_s(N)
    X0.phi[:] = 4. * fft.idstn(phiv, type=1)
    solver = LCSolve(outdir='.', conf=c, N=N, x0=X0, load_file=False,
                     verbose=not args.silent)
    print("search for radial starts.", datetime.datetime.now())
    if not exists("radial_temp.npy"):
        # coarse smoothing, takes >30min
        # skip this step if temporary file has been saved
        solver.solve(method='Q-gd', maxiter=600, eta=1e-4, tol=1e-4,
                     bb=True, verbose=not args.silent)
        solver.solve(method='gd', metric="h1", maxiter=20000, eta=5e-4, tol=1e-4,
                     bb=True, verbose=not args.silent)  # H1 grad descent
        solver.snapshot("radial_temp")
    else:
        solver.X = load_lc("radial_temp.npy", resize=N)
    # It seems that radial is a saddle point between ring and split-core
    # So we use Newton method
    solver.solve(method="newton", metric="l2", maxiter=100, eta=0.8, tol=3e-8,
                 maxsubiter=100, gmres_restart=40, subtol=0.1, verbose=True)
    solver.snapshot("radial%d" % (N + 1))
    print("radial found", datetime.datetime.now())

if args.name == "tactoid":
    # Testified bootstrap of tactoid (volume=0.1)
    r = np.sqrt((xx - .5)**2 + (yy - .5)**2 + 0.9 * (zz - .5)**2)
    phiv = (np.tanh((0.287941 - r) / 0.01) + 1)
    X0 = LCState_s(N)
    c.lam = 2e-6
    X0.phi[:] = 4. * fft.idstn(phiv, type=1)
    solver = LCSolve(outdir='.', conf=c, N=N, x0=X0, load_file=False, verbose=True)
    print("search for tactoid starts.", datetime.datetime.now())
    solver.solve(method='Q-gd', metric="l2", maxiter=200, eta=1e-4, tol=1e-8,
                 bb=True, verbose=not args.silent)
    solver.solve(method='gd', metric="l2", maxiter=20000, eta=1e-4, tol=1e-8,
                 bb=True, verbose=not args.silent)
    solver.snapshot("tactoid%d" % (N + 1))
    print("tactoid found", datetime.datetime.now())

if args.name == "pointy":
    # Testified bootstrap of a more perfect tactoid (volume=0.1)
    r = np.sqrt((xx - .5)**2 + (yy - .5)**2 + 0.9 * (zz - .5)**2)
    phiv = (np.tanh((0.287941 - r) / 0.01) + 1)
    X0 = LCState_s(N)
    c.lam = 1.5e-5
    X0.phi[:] = 4. * fft.idstn(phiv, type=1)
    solver = LCSolve(outdir='.', conf=c, N=N, x0=X0, load_file=False,
                     verbose=not args.silent)
    print("search for pointy tactoid starts.", datetime.datetime.now())
    solver.solve(method='Q-gd', metric="l2", maxiter=200, eta=1e-4, tol=1e-8,
                 bb=True, verbose=not args.silent)
    solver.solve(method='gd', metric="l2", maxiter=20000, eta=1e-4, tol=1e-8,
                 bb=True, verbose=not args.silent)
    solver.snapshot("pointy%d" % (N + 1))
    print("pointy tactoid found", datetime.datetime.now())

if args.name == "twodrops":
    # Testified bootstrap of two droplets (volume sum=0.1)
    r = np.sqrt((xx - .3)**2 + (yy - .3)**2 + (zz - .3)**2)
    d1 = .5 * (np.tanh((.228539 - r) / 0.02) + 1)
    d2 = .5 * (np.tanh((.228539 - r[::-1, ::-1, ::-1]) / 0.02) + 1)
    X0 = LCState_s(N)
    X0.phi[:] = 8. * fft.idstn(d1 + d2, type=1)
    print("search for two droplets starts.", datetime.datetime.now())
    solver = LCSolve(outdir='.', conf=c, N=N, x0=X0, load_file=False,
                     verbose=not args.silent)
    solver.solve(method='Q-gd', maxiter=600, metric="l2", eta=1e-4, tol=1e-8,
                 bb=True, verbose=not args.silent)
    solver.solve(method='newton', maxiter=40, metric="l2", eta=0.8, tol=1e-8,
                 maxsubiter=100, gmres_restart=40, subtol=0.1, verbose=not args.silent)
    print(F.energy(solver.X))
    solver.snapshot(fname="twodrops48")
    print("two droplets found", datetime.datetime.now())

print_profile()
