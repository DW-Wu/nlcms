"""Test different parameters"""

from nlc_solve import *


def test_A_and_lam(x0, c0, outdir, name):
    c = LCConfig(**c0)
    solver = LCSolve(outdir=outdir, conf=c0, N=x0.N, x0=x0, load_file=False, verbose=False)
    As = [-866, -1733, -2600, -3466, -4333, -5200]
    # As = [-866, -2600, -4333]
    lams = [5, 7, 9, 11, 13, 15]
    # lams = [5, 15]
    flags = np.zeros([len(As), len(lams)])
    for i, A in enumerate(As):
        solver.X = resize_lc(x0, x0.N)
        for j, lam in enumerate(lams):
            c.A = A
            c.lam = lam
            F = LCFunc_s(**c)
            solver.solve_gf(F, metric="l2", maxiter=300, eta=1e-3, tol=1e-3)
            solver.solve_newton(F, metric="l2", maxiter=40, tol=1e-5, eta=0.1,
                                damp_threshold=0.3, maxsubiter=100,
                                gmres_restart=40, subtol=0.1, verbose=False)
            flags[i, j] = solver.flag
            if solver.flag == 1:
                print("A=%d\tlam=%d\tsuccessful" % (A, lam))
                solver.snapshot("%s_A=%d_lam=%d" % (name, A, lam))
            elif solver.flag == 0:
                print("A=%d\tlam=%d\tnot convergent" % (A, lam))
                solver.X.x[:] = x0.x[:]  # back to initial state
            else:  # flag==-1, Newton not convergent
                print("A=%d\tlam=%d\tblows up" % (A, lam))
                solver.X.x[:] = x0.x[:]
    return flags


if __name__ == "__main__":
    c = load_lc_config("radial.json")
    c = LCConfig(A=-866, lam=5, v0=0.1,
                 eps=0.005, omega=20, wp=1, wv=0.5)  # same
    F = LCFunc_s(**c)
    N = 63
    Nvec = 1500282
    # xx = np.arange(1, N + 1) / (N + 1)
    # xx, yy, zz = np.meshgrid(xx, xx, xx, indexing='ij')
    x0rd = load_lc("radial64.npy")
    x0rn = load_lc("ring64.npy")
    x0tc = load_lc("tactoid64.npy")

    if len(sys.argv) < 2 or sys.argv[1] == "radial":
        print("Search for radial states")
        print(test_A_and_lam(x0rd, c, "out/test_rd", "radial"))
    elif sys.argv[1] == "ring":
        print("Search for ring states")
        print(test_A_and_lam(x0rn, c, "out/test_rn", "ring"))
    elif sys.argv[1] == "tactoid":
        print("Search for tactoid states")
        print(test_A_and_lam(x0tc, c, "out/test_tc", "tactoid"))
