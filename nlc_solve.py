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
                    help="Initial value in .npy file (overrides -r option)")
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


class LCSolve:
    def __init__(self, outdir, conf, N, x0=None, load_file=True, verbose=False):
        # Check output directory
        if not exists(outdir):
            os.makedirs(outdir)
        # Get configuration
        self.outdir = outdir
        self.conf = LCConfig(**conf)
        if verbose:
            # Show functional
            FF = LCFunc_s(**conf)
            FF.reset_params(show=True, **conf)
        # State
        if x0 is not None:
            # user-defined initial value (highest priority)
            self.X = resize_lc(x0, N)
        elif load_file and exists(join(outdir, "solution.npy")):
            # Load existing file
            if verbose:
                print("Loading existing solution...")
            self.X = load_lc(join(outdir, "solution.npy"), resize=N)
        else:
            if verbose:
                print("Starting iteration from scratch...")
            # Initialize spherical state
            self.X = LCState_s(N)
            xx = np.arange(1, N + 1) / N
            xx, yy, zz = np.meshgrid(xx, xx, xx, indexing='ij')
            r = np.sqrt((xx - .5)**2 + (yy - .5)**2 + (zz - .5)**2)
            phiv = (np.tanh(((3 * conf.v0 / 4 / np.pi)**(1 / 3) - r) / 0.04) + 1)
            self.X.phi[:] = 4. * fft.idstn(phiv, type=1)
        self.X.proj_phi(self.conf.v0)

        # Storage needed in iteration
        self.fvec = []
        self.flag = 0

    def solve(self, method="gd", **kwargs):
        """Solve for minimizer (start from any state)
        Different methods requires different kw arguments"""
        # Initialize
        N = self.X.N
        FF = LCFunc_s(**self.conf)
        # FF.reset_conf(self.conf)
        FF.get_aux(N)
        G = FF.grad(self.X, proj=True)
        gnorm = norm(G.x)
        self.fvec = [FF.energy(self.X)]
        if gnorm < kwargs.get('tol', 1e-8):
            if kwargs.get('verbose', False):
                print("Solution found! |g| =", gnorm)
            self.flag = 1
            return 0
        FF.export_conf(join(self.outdir, "conf.json"))

        if method == "gd":
            # Simple gradient descent
            itn = self.solve_gd(FF, Q_only=False, **kwargs)
        elif method == "Q-gd":
            # Gradient descent, but only in Q direction
            itn = self.solve_gd(FF, Q_only=True, **kwargs)
        elif method == "gf":
            # Gradient flow using implicit diffusion operator
            itn = self.solve_gf(FF, **kwargs)
        return itn

    def snapshot(self, fname="solution"):
        save_lc(join(self.outdir, fname + ".npy"), self.X)

    def solve_gd(self, FF, Q_only=False,
            maxiter=2000,
            eta=1e-3,
            tol=1e-8,
            verbose=False,
            bb=False):
        X = self.X
        s = np.zeros_like(X.x)
        y = np.zeros_like(X.x)
        self.flag = 0
        for k in (tqdm(range(maxiter), desc="GD") \
                if verbose else maxiter):
            G = FF.grad(X, proj=True)
            gnorm = norm(G.q) if Q_only else norm(G.x)
            if gnorm < tol:
                self.flag = 1
                break
            a = eta
            if bb and k > 0:
                y[:] = G.x - y
                if Q_only:
                    y[5 * N**3:] = 0.
                a = np.abs(np.dot(s, y) / np.dot(y, y))
                # if a < -1e-2:
                #     a = eta  # Keep step length positive
            # Descent
            s[:] = X.x
            X.q[:] -= a * G.q
            if not Q_only:
                X.phi[:] -= a * G.phi
                FF.project(X, FF.v0)
            self.fvec.append(FF.energy(X))
            if np.any(np.isnan(X.x)):
                self.flag = -1
                if verbose:
                    print("NAN @ itno.", k)
                break
            if bb:
                s[:] = X.x[:] - s
                y[:] = G.x
        if verbose and self.flag == 0:
            print("Iteration failed to converge, |g| =", gnorm)
        return k  # return iteration number. k large implies state change

    def solve_gf(self, FF,
            maxiter=2000,
            eta=1e-3,
            tol=1e-8,
            maxsubiter=50,
            subtol=0.1,
            verbose=False):
        X = self.X
        Xp = LCState_s(X.N)
        for t in (tqdm(range(maxiter), desc="GF") \
                if verbose else range(maxiter)):
            G_nonlin = FF.grad(X, part=1, proj=True)
            Xp.x[:] = X.x
            D = FF.diffusion(Xp, proj=True)

            # An implicit step involves solving the anchoring diffusion operator
            # We use GMRES
            Y = (X.x - eta * G_nonlin.x).ravel()
            IpD = LinearOperator(dtype=float, shape=(6 * N**3, 6 * N**3),
                                 matvec=lambda v: v + eta * (D @ v))
            # Solve implicit equation in phi using GMRES (very loose conditions)
            x_new, _ = gmres(IpD, Y, Xp.x,
                             rtol=subtol * eta, restart=20, maxiter=maxsubiter)
            X.x[:] = x_new
            FF.project(X, FF.v0)
            self.fvec.append(FF.energy(X))
            if np.any(np.isnan(X.x)):
                self.flag = -1
                if verbose:
                    print("NAN @ itno.", t)
                break
            gnorm = norm(X.x - Xp.x) / eta
            if gnorm < tol:
                self.flag = 1
                if verbose:
                    print("Iteration successful @ itno.", t, ", |dx/dt| =", gnorm)
                break
        if verbose and self.flag == 0:
            print("Iteration failed to converge, |dx/dt| =", gnorm)
        return t  # return iteration number


if __name__ == "__main__":
    if not exists(args.output):
        os.mkdir(args.output)

    # a=3
    N = int(args.num_sines)  # default 31
    np.random.seed(20240909)
    # Default config for ring state
    c0 = LCConfig(A=-1500, lam=2e-7, v0=0.1,
                  eps=0.005, omega=20, wp=1, wv=0.5)
    if args.config is not None:
        c0.update(load_lc_config(args.config))
    FF = LCFunc_s(**c0)
    # Initial value
    X = None
    if args.init is not None:
        # user-defined initial value (highest priority)
        if not args.silent:
            print("Loading initial value from file...")
        X = load_lc(args.init, resize=N)

    # Solve with scipy's routines
    # x = fmin_cg(FF.energy_vec, X.x, fprime=FF.grad_vec,
    #             args=(N, True),
    #             gtol=1e-6,
    #             maxiter=10000)
    # X = LCState_s(N, x)
    # FF.project(X, FF.v0)

    solver = LCSolve(outdir=args.output, conf=c0, N=N, x0=X, load_file=not args.restart)
    # Smoothen with gradient flow first
    solver.solve(method='gf', maxiter=200, eta=1e-4, tol=1e-6,
                 maxsubiter=50, subtol=0.05, verbose=False)
    solver.solve(method='gd', maxiter=int(args.maxiter),  # default 2000
                 eta=float(args.eta),  # default 1e-6
                 tol=float(args.tol) * np.sqrt(6 * N**3),  # default 1e-8
                 verbose=not args.silent,  # default True
                 bb=True)
    solver.snapshot()

    plt.plot(solver.fvec)
    plt.title("Energy in gradient flow")
    plt.savefig(join(args.output, "energy.pdf"))
    if not args.silent:
        print("Energy = %.6f" % FF.energy(solver.X))

    # Export to matlab format
    if args.matlab:
        X_v = solver.X.sine_trans(sz=63)
        xxx = np.arange(1, 64) / 64.
        xx, yy, zz = np.meshgrid(xxx, xxx, xxx, indexing='ij')
        scipy.io.savemat(join(args.output, "solution.mat"),
                         {"xx": xx.ravel(), "yy": yy.ravel(), "zz": zz.ravel(),
                          "q1": X_v.q1.ravel(),
                          "q2": X_v.q2.ravel(),
                          "q3": X_v.q3.ravel(),
                          "q4": X_v.q4.ravel(),
                          "q5": X_v.q5.ravel(),
                          "phi": X_v.phi.ravel()}, do_compression=True)

    if args.post == "plot":
        # Leave plotting to the other script
        subprocess.run([PY_EXE, "nlc_plot.py", join(args.output, "solution.npy"),
                        "-N=127", "-o", args.output])
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
