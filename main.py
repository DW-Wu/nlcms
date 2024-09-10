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
                    help="Initial value as .npy file")

if __name__ == "__main__":
    # Parse args first. Avoid loading modules if -h flag is passed
    args = parser.parse_args()

import sys, os
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize as optimize

from nlc_func import *

# python executable name (depends on platform)
PY_EXE = "./Scripts/python" if sys.platform == "win32" else "python3"

if __name__ == "__main__":
    OUTD = join(os.path.dirname(sys.argv[0]), args.output)
    if not os.path.exists(OUTD):
        os.mkdir(OUTD)

    # a=3
    N = 31
    np.random.seed(20240909)
    FF = LCFunc_s()
    # Default config for radial state
    c0 = LCConfig(A=-1500, lam=1e-6, vol=0.2,
                  alpha=1. / 64, wv=10, wp=1, wa=5)
    if args.config is not None:
        c0.update(load_lc_config(args.config))
    FF.reset_conf(c0, show=True)
    FF.get_aux(N)
    FF.export_conf(join(OUTD, "conf.json"))
    # Initial value
    if args.init is not None:
        # user-defined initail value (highest priority)
        print("Loading initial value from file...")
        X = load_lc(args.init)
    elif os.path.exists(join(OUTD, "solution.npy")) and not args.restart:
        # solution obtained last time
        print("Loading existing solution...")
        X = load_lc(join(OUTD, "solution.npy"))
    else:
        # reinitialize
        print("Starting iteration from scratch...")
        X = LCState_s(N)
        xx = np.arange(1, N + 1) / N
        # spherical initial value
        xx, yy, zz = np.meshgrid(xx, xx, xx, indexing='ij')
        r = np.sqrt((xx - .5) ** 2 + (yy - .5) ** 2 + (zz - .5) ** 2)
        X.phi[:] = 8. * fft.idstn(np.tanh((r - (3 * FF.v0 / 4 / np.pi) ** (1 / 3)) / 0.05), type=1)
        # X.phi[0, 0, 0] = c0['vol']/FF.aux.ios3[0,0,0]
    FF.project(X, c0['vol'])

    # Solve with scipy's routines
    # x=optimize.fmin_cg(FF.energy_vec,X.x,fprime=FF.grad_vec,args=(N,True),gtol=1e-6,maxiter=20000,c1=-1e-6)
    # print(norm(FF.grad_vec(x,N)))
    # X=LCState_s(N,x)
    # FF.project(X,c0['vol'])

    # solve for state
    X, flag = FF.grad_descent(X, maxiter=100, eta=2e-6, tol=1e-6,
                              bb=False, verbose=0)
    if flag <= 0:
        X, flag, fvec = FF.grad_descent(X, maxiter=9000, eta=2e-6, tol=1e-6,
                                        bb=True, verbose=1, inspect=True)
    print("Energy = %.6f" % FF.energy(X))
    save_lc(join(OUTD, "solution.npy"), X)
    # print(np.sum(FF.aux.ios3*X.phi))

    # Plot
    s = 63
    X_v = X.sine_trans(sz=s)
    xxx = np.arange(1, s + 1) / (s + 1)
    xx, yy, zz = np.meshgrid(xxx, xxx, xxx, indexing='ij')
    # Export to matlab format
    scipy.io.savemat(join(OUTD, "solution.mat"),
                     {"xx": xx.ravel(), "yy": yy.ravel(), "zz": zz.ravel(),
                      "q1": X_v.q1.ravel(),
                      "q2": X_v.q2.ravel(),
                      "q3": X_v.q3.ravel(),
                      "q4": X_v.q4.ravel(),
                      "q5": X_v.q5.ravel(),
                      "phi": X_v.phi.ravel()}, do_compression=True)

    # Leave plotting to the other script
    import subprocess

    subprocess.run([PY_EXE, "nlc_plot.py", join(OUTD, "solution.npy"),
                    "-N", str(s), "-o", OUTD])
