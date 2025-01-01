"""Plot landscape near radial solution"""

from nlc_solve import *
from scipy.interpolate import RectBivariateSpline


def biaxiality(q1, q2, q3, q4, q5):
    """tr(Q^3)^2/tr(Q^2)^3, which is between 0 and 1/6"""
    return 1 - 6 * trace_Q3(q1, q2, q3, q4, q5)**2 / (trace_Q2(q1, q2, q3, q4, q5)**3 + 1e-14)


if __name__ == "__main__":
    c = load_lc_config("ring.json")
    c = LCConfig(A=-1500, lam=2e-7, v0=0.1,
                 eps=0.005, omega=20, wp=1, wv=0.5)  # same
    F = LCFunc_s(**c)
    N = 47
    Nvec = 622938
    xx = np.arange(1, N + 1) / (N + 1)
    xx, yy, zz = np.meshgrid(xx, xx, xx, indexing='ij')

    x = load_lc("radial48.npy", resize=N)
    x_rn = load_lc("ring48.npy")
    x_sc = load_lc("score48.npy")
    print("E(ring): %f, E(radial) = %f, E(splitcore) = %f"
          % (F.energy(x_rn), F.energy(x), F.energy(x_sc)))

    # Plot landscape around radial
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    w1 = x_sc.x - x.x
    w1 /= norm(w1)
    w2 = x_rn.x - x.x
    w2 -= np.dot(w2, w1) * w1
    w2 /= norm(w2)
    x1, y1 = np.dot(x_sc.x - x.x, w1), np.dot(x_sc.x - x.x, w2)
    x2, y2 = np.dot(x_rn.x - x.x, w1), np.dot(x_rn.x - x.x, w2)
    xlim = (-.08, .1)
    ylim = (-.05, .07)
    xx = np.linspace(*xlim, 17)
    yy = np.linspace(*ylim, 17)
    x_loc, y_loc = np.meshgrid(xx, yy, indexing="ij")
    E_loc = np.zeros_like(x_loc)
    for i, (X, Y) in enumerate(zip(x_loc.ravel(), y_loc.ravel())):
        xs = x.x + X * w1 + Y * w2
        E_loc.ravel()[i] = F.energy_vec(xs, N)
    # Interpolate and plot contours
    interp = RectBivariateSpline(xx, yy, E_loc)
    xxf = np.linspace(*xlim, 301)
    yyf = np.linspace(*ylim, 201)
    Xf, Yf = np.meshgrid(xxf, yyf, indexing="ij")
    E_fine = interp(xxf, yyf)  # returns grid function with `ij` ordering
    Er = np.tanh((E_fine - np.min(E_fine)) * 400)
    img = ax.imshow(Er.T, cmap='RdBu', interpolation="bicubic",
                    extent=xlim + ylim, origin="lower")
    ax.contour(Xf, Yf, Er)
    ax.scatter(0, 0, label="radial")
    ax.scatter(x1, y1, label="spilt-core")
    ax.scatter(x2, y2, label="ring")
    ax.legend()
    ax.set(title="Rescaled energy $E_r=\\tanh(400(E-E_{\min}))$",
           xticks=[], yticks=[], xlim=xlim, ylim=ylim)
    plt.colorbar(img, ax=ax)
    # plt.show()
    fig.savefig("out/landscape.pdf")

    # Compute eigenvalue
    H = F.hess(x, proj="l2")
    if exists("radial48_eig1.npy"):
        V = np.load("radial48_eig1.npy")
        lam = np.dot(V, H @ V)
    else:
        lam, V = lobpcg(H, np.eye(Nvec, 1), tol=1e-6, maxiter=4000,
                        largest=False, verbosityLevel=0)
        np.save("radial48_eig1.npy", V.ravel())
    print("Smallest eigenvalue at radial:", lam)
