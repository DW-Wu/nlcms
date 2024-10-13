from argparse import ArgumentParser

parser = ArgumentParser(prog="nlcms",
                        description="Diffuse-interface LdG model")
parser.add_argument('-c', '--config', action="store", default=None,
                    help="Config file (see LCConfig definition for format)")
parser.add_argument('-o', '--output', action="store", default='out',
                    help="Output directory")
parser.add_argument('-r', '--restart', action="store_true", default=False,
                    help="Whether to restart iteration (otherwise use stored initial value)")
parser.add_argument('-i', '--init', action="store", default=None,
                    help="Initial value as .npy file (overrides -r option)")
parser.add_argument('-s', '--silent', action="store_true", default=False,
                    help="Silent run (no printed messages)")
parser.add_argument('-N', '--num-sines', action="store", default=31, type=int,
                    help="Number of sine functions along each axis")
parser.add_argument('--maxiter', action='store', default=2000, type=int,
                    help="Maximum iteration number")
parser.add_argument('--eta', action='store', default=1e-6, type=float,
                    help="Gradient descent step length (learning rate)")
parser.add_argument('--tol', action='store', default=1e-8, type=float,
                    help="Gradient norm tolerance (relative to dimension)")
parser.add_argument('--matlab', action='store_true', default=False,
                    help="Export .mat file of state variables")
parser.add_argument('--post', action='store', default="plot", choices=["plot", "eigen", "none"],
                    help="Postprocessing routine: (plot graphs | compute eigenvalues | none)")

if __name__ == "__main__":
    # Parse args first. Avoid loading modules if -h flag is passed
    args = parser.parse_args()

import sys
import os
from os.path import join, exists
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import fmin_cg
from scipy.sparse.linalg import lobpcg, gmres
from tqdm import tqdm

from nlc_func import *

# python executable name (depends on platform)
PY_EXE = "./Scripts/python" if sys.platform == "win32" else "python3"


def solve_gd(FF: LCFunc_s, X0: LCState_s, Q_only=False,
             maxiter=10000, eta=1e-3, tol=1e-8,
             bb=False, verbose=0, inspect=False):
    """Solve for minimizer with gradient descent"""
    X = LCState_s(X0.N)
    X.x[:] = X0.x
    if inspect:
        fvec = np.zeros(maxiter)
    flag = 0
    Xp = np.zeros_like(X.x)
    Gp = np.zeros_like(X.x)
    s = np.zeros_like(X.x)
    for k in (tqdm(range(maxiter), desc="GD") if verbose else range(maxiter)):
        G = FF.grad(X, proj=True)
        gnorm = norm(G.q) if Q_only else norm(G.x)
        if verbose >= 2:
            print("Gradient norm @ itno. %d: %.3e" % (k, gnorm))
        if gnorm < tol:
            flag = 1
            break
        a = eta
        if bb and k > 0:
            y = G.x - Gp
            if Q_only:
                y[5 * X0.N**3:] = 0.
            a = np.abs(np.dot(s, y) / np.dot(y, y))
            # if a < -1e-2:
            #     a = eta  # Keep step length positive
        # print(k,"::", a, gnorm)
        # Descent
        Xp[:] = X.x
        X.q[:] -= a * G.q
        if not Q_only:
            X.phi[:] -= a * G.phi
            FF.project(X, FF.v0)
        if np.any(np.isnan(X.x)):
            flag = -1
            break
        if bb:
            s[:] = X.x[:] - Xp
            Gp[:] = G.x
        if inspect:
            fvec[k] = FF.energy(X)
    if verbose:
        if flag == 0:
            print("Iteration failed to converge, |g| =", gnorm)
        elif flag == 1:
            print("Iteration successful @ itno.", k, ", |g| =", gnorm)
        elif flag == -1:
            print("NAN @ itno.", k)
    if inspect:
        return X, flag, fvec[0:k]
    return X, flag


def solve_gf(FF: LCFunc_s, X0: LCState_s,
             maxiter=10000,
             dt=1e-3,
             tol=1e-8,
             maxsubiter=50,
             subtol=0.1,
             verbose=0,
             inspect=False):
    """Solve with implicit gradient flow"""
    N = X0.N
    X = LCState_s(N)
    X.x[:] = X0.x
    if inspect:
        fvec = np.zeros(maxiter)
    flag = 0
    Xp = LCState_s(N)

    for t in (tqdm(range(maxiter), desc="GF") if verbose else range(maxiter)):
        G_nonlin = FF.grad(X, part=1, proj=True)
        Xp.x[:] = X.x
        D = FF.diffusion(Xp, proj=True)
        if inspect:
            fvec[t] = FF.energy(X)
        # An implicit step involves solving the anchoring diffusion operator
        # We use GMRES
        Y = (X.x - dt * G_nonlin.x).ravel()
        IpD = LinearOperator(dtype=float, shape=(6 * N**3, 6 * N**3),
                             matvec=lambda v: v + dt * (D @ v))
        # Solve implicit equation in phi using GMRES (very loose conditions)
        x_new, _ = gmres(IpD, Y, Xp.x,
                         rtol=subtol * dt, restart=20, maxiter=maxsubiter)
        X.x[:] = x_new
        FF.project(X, FF.v0)
        if np.any(np.isnan(X.x)):
            flag = -1
            break
        gnorm = norm(X.x - Xp.x) / dt
        if gnorm < tol:
            flag = 1
            break
    if verbose:
        if flag == 0:
            print("Iteration failed to converge, |dx/dt| =", gnorm)
        elif flag == 1:
            print("Iteration successful @ itno.", t, ", |dx/dt| =", gnorm)
        elif flag == -1:
            print("NAN @ itno.", t)
    if inspect:
        return X, flag, fvec[0:t]
    return X, flag


if __name__ == "__main__":
    OUTD = join(os.path.dirname(sys.argv[0]), args.output)
    if not exists(OUTD):
        os.mkdir(OUTD)

    # a=3
    N = int(args.num_sines)  # default 31
    np.random.seed(20240909)
    FF = LCFunc_s()
    # Default config for ring state
    c0 = LCConfig(A=-1500, lam=2e-7, v0=0.1,
                  eps=0.005, omega=20, wp=1, wv=0.5)
    if args.config is not None:
        c0.update(load_lc_config(args.config))
    FF.reset_conf(c0, show=not args.silent)
    FF.get_aux(N)
    FF.export_conf(join(OUTD, "conf.json"))
    # Initial value
    if args.init is not None:
        # user-defined initial value (highest priority)
        if not args.silent:
            print("Loading initial value from file...")
        X = load_lc(args.init, resize=N)
    elif exists(join(OUTD, "solution.npy")) and not args.restart:
        # solution obtained last time
        if not args.silent:
            print("Loading existing solution...")
        X = load_lc(join(OUTD, "solution.npy"), resize=N)
    else:
        # reinitialize
        if not args.silent:
            print("Starting iteration from scratch...")
        X = LCState_s(N)
        xx = np.arange(1, N + 1) / N
        # spherical initial value
        xx, yy, zz = np.meshgrid(xx, xx, xx, indexing='ij')
        r = np.sqrt((xx - .5)**2 + (yy - .5)**2 + (zz - .5)**2)
        phiv = (np.tanh(((3 * FF.v0 / 4 / np.pi)**(1 / 3) - r) / 0.04) + 1)
        X.phi[:] = 4. * fft.idstn(phiv, type=1)
        # X.phi[0, 0, 0] = c0['vol']/FF.aux.ios3[0,0,0]
    FF.project(X, FF.v0)

    # Solve with scipy's routines
    # x = fmin_cg(FF.energy_vec, X.x, fprime=FF.grad_vec,
    #             args=(N, True),
    #             gtol=1e-6,
    #             maxiter=10000)
    # X = LCState_s(N, x)
    # FF.project(X, FF.v0)

    # solve for state
    g = FF.grad(X)
    g.proj_phi(0)
    if norm(g.x) > .1:
        # Pre-process by minimizing on Q first
        X, _ = solve_gd(FF, X, Q_only=True,
                        maxiter=1000,
                        eta=1e-5,
                        tol=float(args.tol) * np.sqrt(6 * N**3),
                        bb=True, verbose=0)
        # Use GF to smoothen
        X, _, fvec = solve_gf(FF, X, maxiter=100, dt=1e-4, tol=1e-8, verbose=1, inspect=True)
        plt.plot(fvec)
        plt.title("Energy in gradient flow")
        plt.savefig(join(OUTD, "energy.pdf"))
    X, _ = solve_gd(FF, X,
                    maxiter=int(args.maxiter),  # default 2000
                    eta=float(args.eta),  # default 1e-6
                    tol=float(args.tol) * np.sqrt(6 * N**3),  # default 1e-8
                    bb=True,
                    verbose=not args.silent,  # default True
                    inspect=False)
    if not args.silent:
        print("Energy = %.6f" % FF.energy(X))
    save_lc(join(OUTD, "solution.npy"), X)
    # print(np.sum(FF.aux.ios3*X.phi))

    # Export to matlab format
    if args.matlab:
        X_v = X.sine_trans(sz=63)
        xxx = np.arange(1, 64) / 64.
        xx, yy, zz = np.meshgrid(xxx, xxx, xxx, indexing='ij')
        scipy.io.savemat(join(OUTD, "solution.mat"),
                         {"xx": xx.ravel(), "yy": yy.ravel(), "zz": zz.ravel(),
                          "q1": X_v.q1.ravel(),
                          "q2": X_v.q2.ravel(),
                          "q3": X_v.q3.ravel(),
                          "q4": X_v.q4.ravel(),
                          "q5": X_v.q5.ravel(),
                          "phi": X_v.phi.ravel()}, do_compression=True)

    if args.post == "plot":
        # Leave plotting to the other script
        subprocess.run([PY_EXE, "nlc_plot.py", join(OUTD, "solution.npy"),
                        "-N=127", "-o", OUTD])
    elif args.post == "eigen":
        H = FF.hess(X)
        lam, V = lobpcg(H, np.eye(6 * N**3, 3),
                        Y=np.asmatrix(np.hstack([np.zeros(5 * N**3), FF.aux.ios3.ravel()])).T,
                        maxiter=int(args.maxiter),
                        tol=1e-6, largest=False,
                        verbosityLevel=0)
        print("Smallest eigenvalues:", lam)
        # save_lc("eig1.npy",LCState_s(N,V[:,0],copy=True))
        # save_lc("eig2.npy",LCState_s(N,V[:,1],copy=True))
