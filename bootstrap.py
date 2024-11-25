"""Bootstraping the stable states from simple initial values"""

from nlc_func import *
from nlc_solve import LCSolve
import datetime

c = load_lc_config("ring.json")
c = LCConfig(A=-1500, lam=2e-7, v0=0.1,
             eps=0.005, omega=20, wp=1, wv=0.5)  # same
F = LCFunc_s(**c)
N = 47
xx = np.arange(1, N + 1) / (N + 1)
xx, yy, zz = np.meshgrid(xx, xx, xx, indexing='ij')

# Testified bootstrap of ring (volume=0.1)
# start from an oblate sphere
r = np.sqrt((xx - .5)**2 + (yy - .5)**2 + 1.1 * (zz - .5)**2)
phiv = (np.tanh((0.287941 - r) / 0.04) + 1)
X0 = LCState_s(N)
X0.phi[:] = 4. * fft.idstn(phiv, type=1)
solver = LCSolve(outdir='.', conf=c, N=N, x0=X0, load_file=False, verbose=True)
print("search for ring starts", datetime.datetime.now())
solver.solve(method='gf', maxiter=200, eta=1e-4, tol=1e-6,
             maxsubiter=50, subtol=0.05, verbose=True)
solver.solve(method='gd', maxiter=10000, eta=1e-4, tol=1e-8,
             bb=True, verbose=True)
solver.snapshot(fname="ring48")
print("ring is found", datetime.datetime.now())

# Testified bootstrap of split-core (volume=0.1)
# Start fro ma sprolate sphere
r = np.sqrt((xx - .5)**2 + (yy - .5)**2 + 0.8 * (zz - .5)**2)
phiv = (np.tanh((0.287941 - r) / 0.04) + 1)
X0[:] = 0
X0.phi[:] = 4. * fft.idstn(phiv, type=1)
X0.proj_phi(c.v0)
solver.X.x[:] = X0.x[:]
print("search for split-core starts", datetime.datetime.now())
# Stabilize Q
solver.solve(method='Q-gd', maxiter=600, eta=1e-4, tol=1e-8, bb=True, verbose=True)
solver.solve(method='gf', maxiter=400, eta=1e-4, tol=1e-6,
             maxsubiter=50, subtol=0.05, verbose=True)
solver.solve(method='gd', maxiter=20000, eta=1e-4, tol=1e-8,
             bb=True, verbose=True)
solver.snapshot(fname="score")
print("split-core is found", datetime.datetime.now())

# Testified bootstrap of radial solution (volume=0.1)
r = np.sqrt((xx - .5)**2 + (yy - .5)**2 + (zz - .5)**2)
phiv = (np.tanh((0.287941 - r) / 0.01) + 1)
X0[:] = 0
X0.phi[:] = 4. * fft.idstn(phiv, type=1)
X0.proj_phi(c.v0)
solver.X.x[:] = X0.x[:]
print("search for radial starts.", datetime.datetime.now())
solver.solve(method='Q-gd', maxiter=200, eta=1e-4, tol=1e-8, bb=True, verbose=True)
solver.solve(method='gd', maxiter=20000, eta=1e-4, tol=1e-8, bb=True, verbose=True)
solver.snapshot("radial")
print("radial found", datetime.datetime.now())

# Testified bootstrap of tactoid (volume=0.1)
r = np.sqrt((xx - .5)**2 + (yy - .5)**2 + 0.9 * (zz - .5)**2)
phiv = (np.tanh((0.287941 - r) / 0.01) + 1)
X0[:] = 0
X0.phi[:] = 4. * fft.idstn(phiv, type=1)
X0.proj_phi(c.v0)
solver.X.x[:] = X0.x[:]
c.lam = 2e-6
print("search for tactoid starts.", datetime.datetime.now())
solver.solve(method='Q-gd', maxiter=200, eta=1e-4, tol=1e-8, bb=True, verbose=True)
solver.solve(method='gd', maxiter=20000, eta=1e-4, tol=1e-8, bb=True, verbose=True)
solver.snapshot("tactoid48")
print("tactoid found", datetime.datetime.now())
# Use this solution as initial value, and change lam to 2e-5
# You get an even pointier tactoid
