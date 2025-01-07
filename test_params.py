"""Test different parameters"""

from nlc_func import *
import sys
from nlc_solve import LCSolve
from argparse import ArgumentParser


def test_param(c0, param, param_list, x0,
               maxiter=2000,
               outdir="test",
               prefix="",
               verbose=False):
    c = LCConfig(**c0)
    solver = LCSolve(outdir=outdir, conf=c0, N=x0.N, x0=x0, load_file=False,
                     verbose=verbose)
    if prefix:
        prefix += '_'
    for p in param_list:
        c[param] = p
        F = LCFunc_s(**c)
        solver.solve_gd(F, metric="l2", maxiter=maxiter, eta=2e-4, tol=1e-6,
                        bb=True, verbose=verbose)
        # Use Newton method to accelerate convergence
        solver.solve_newton(F, metric="l2", maxiter=maxiter // 100, tol=1e-6,
                            eta=0.1, damp_threshold=0.9,
                            maxsubiter=50, gmres_restart=40, subtol=0.01,
                            verbose=verbose)
        print(prefix, param, p)
        solver.snapshot(prefix + param + "=%.1e" % p)
        if solver.flag != 1:
            print("Iteration fails")


parser = ArgumentParser(prog="test_params")
parser.add_argument("-s", "--silent", action="store_true", default=False)


if __name__ == "__main__":
    args = parser.parse_args()
    N = 47
    c = LCConfig(A=-1500, lam=2e-7, v0=0.1,
                 eps=0.005, omega=20, wp=1, wv=0.5)
    F = LCFunc_s(**c)
    xx = np.arange(1, N + 1) / (N + 1)
    xx, yy, zz = np.meshgrid(xx, xx, xx, indexing='ij')
    r = np.sqrt((xx - .5)**2 + (yy - .5)**2 + (zz - .5)**2)
    phiv = (np.tanh((0.287941 - r) / 0.01) + 1)
    # Prepare a radially symmetric initial value
    if not os.path.exists("radial_temp.npy"):
        X0 = LCState_s(N)
        X0.phi[:] = 4. * fft.idstn(phiv, type=1)
        print("preparing spherical droplet as initial value...")
        solver = LCSolve(outdir='.', conf=c, N=N, x0=X0, load_file=False, verbose=False)
        solver.solve_gd(F, Q_only=True, metric="l2", maxiter=1000, eta=1e-4,
                        tol=1e-5, bb=True, verbose=not args.silent)
        X0 = LCState_s(N, solver.X.x)
        solver.snapshot("radial_temp")
    else:
        X0 = load_lc("radial_temp.npy")

    x1 = load_lc("ring48.npy")
    x2 = load_lc("score48.npy")
    x3 = load_lc("smalltactoid48.npy")
    solver = LCSolve(outdir="out/test", conf=c, N=N, x0=x1,
                     load_file=False, verbose=not args.silent)

    # Adjust lambda
    lam_list = [2e-7, 4e-7, 1e-6, 4e-6, 1e-5]
    test_param(c, "lam", lam_list, x1,
               outdir="out/test", prefix="ring", verbose=not args.silent)
    test_param(c, "lam", lam_list, x2,
               outdir="out/test", prefix="score", verbose=not args.silent)
    test_param(c, "lam", lam_list, x3,
               outdir="out/test", prefix="tactoid", verbose=not args.silent)

    # Adjust A
    A_list = [420, -1500, -3000, -6000]
    test_param(c, "A", A_list, X0, maxiter=5000,
               outdir="out/test", prefix="radial", verbose=not args.silent)

    # Adjust ω
    om_list = [5, 10, 20, 40, 80]
    test_param(c, "omega", om_list, x1,
               outdir="out/test", prefix="ring", verbose=not args.silent)
    test_param(c, "omega", om_list, x2,
               outdir="out/test", prefix="score", verbose=not args.silent)
    test_param(c, "omega", om_list, x3,
               outdir="out/test", prefix="tactoid", verbose=not args.silent)
