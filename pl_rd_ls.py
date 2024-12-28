"""Plot landscape near radial solution"""

from nlc_solve import *
from scipy.interpolate import RectBivariateSpline

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
    w1 = x_sc.x - x.x
    w1 /= norm(w1)
    w2 = x_rn.x - x.x
    w2 -= np.dot(w2, w1) * w1
    w2 /= norm(w2)
    x1, y1 = np.dot(x_sc.x - x.x, w1), np.dot(x_sc.x - x.x, w2)
    x2, y2 = np.dot(x_rn.x - x.x, w1), np.dot(x_rn.x - x.x, w2)
    print("cos<ring - radial, splitcore-radial> =", np.dot(x_rn.x - x.x, x_sc.x - x.x) /
          (norm(x_rn.x - x.x) * norm(x_sc.x - x.x)))
    xlim = (-.08, .1)
    ylim = (-.05, .07)
    xx = np.linspace(*xlim, 17)
    yy = np.linspace(*ylim, 17)
    x_loc, y_loc = np.meshgrid(xx, yy, indexing="ij")
    E_loc = np.zeros_like(x_loc)
    for i, (X, Y) in enumerate(zip(x_loc.ravel(), y_loc.ravel())):
        xs = x.x + X * w1 + Y * w2
        E_loc.ravel()[i] = F.energy_vec(xs, N)
    # Interpolate onto finer grid
    interp = RectBivariateSpline(xx, yy, E_loc)
    xxf = np.linspace(*xlim, 301)
    yyf = np.linspace(*ylim, 201)
    Xf, Yf = np.meshgrid(xxf, yyf)
    E_fine = interp(xxf, yyf)  # returns grid function with `ij` ordering
    print(Xf.shape, E_fine.shape)
    img = plt.imshow(E_loc.T, cmap='RdBu', interpolation="bicubic", extent=xlim + ylim, origin="lower")
    # img = plt.imshow(E_fine.T, cmap='RdBu', extent=xlim + ylim, origin='lower')
    # img = plt.contourf(x_loc, y_loc, E_loc, cmap="RdBu", levels=400)
    plt.scatter([0, x1, x2], [0, y1, y2], c='g')
    plt.text(x1, y1, "split-core", ha="center", va="bottom")
    plt.text(x2, y2, "ring", ha="center", va="bottom")
    plt.text(0, 0, "radial", ha="center", va="bottom")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.colorbar(img)
    # plt.show()
    plt.savefig("out/landscape.pdf")
