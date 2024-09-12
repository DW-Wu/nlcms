"""Computations of the LC state"""
from nlc_state import *
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator


# Some computation on the Q tensor
def Qtimes(q1, q2, q3, q4, q5, v1, v2, v3):
    """Compute Q*v as arrays
    ** The shapes of q1~q5 and v1~v3 must be identical respectively;
    the multiplication between q[1-5] and v[1-3] should be permitted by numpy. **"""
    return (q1 - q3) * v1 + q2 * v2 + q4 * v3, \
           q2 * v1 + (-q1 - q3) * v2 + q5 * v3, \
           q4 * v1 + q5 * v2 + 2 * q3 * v3


def dQtimesv(q1, q2, q3, q4, q5, v1, v2, v3,
             dq1, dq2, dq3, dq4, dq5, dv1, dv2, dv3):
    """Compute variation of Q*v (by Leibniz's rule)"""
    return (q1 - q3) * dv1 + q2 * dv2 + q4 * dv3 \
           + (dq1 - dq3) * v1 + dq2 * v2 + dq4 * v3, \
           q2 * dv1 + (-q1 - q3) * dv2 + q5 * dv3 \
           + dq2 * v1 + (-dq1 - dq3) * v2 + dq5 * v3, \
           q4 * dv1 + q5 * dv2 + 2 * q3 * dv3 \
           + dq4 * v1 + dq5 * v2 + 2 * dq3 * v3


def solve_cubic(a, b, c, largest=False, maxiter=10):
    """Solve cubic equation x^3+ax^2+bx+c=0 for largest/smallest root
    p,q may be vectors so multiple equations can be solved simultaneously"""
    # equivalent to y^2+py+q with y=x+a/3
    p = (3 * b - a ** 2) / 3
    q = (9 * a * b - 27 * c - 2 * a ** 3) / 27
    if largest:
        x = np.maximum(1, np.sqrt(np.abs(p) + np.abs(q))) - a / 3  # x0 > largest root
    else:
        x = -np.maximum(1, np.sqrt(np.abs(p) + np.abs(q))) - a / 3  # x0 < smallest root
    for _ in range(maxiter):
        f = x ** 3 + a * x ** 2 + b * x + c
        if np.max(np.abs(f)) < 1e-10:
            # print('Newton successful')
            break
        x -= (x ** 3 + a * x ** 2 + b * x + c) / (3 * x ** 2 + 2 * a * x + b)
    return x


def trace_Q2(q1, q2, q3, q4, q5):
    return 2 * (q1 * q1 + q2 * q2 + 3 * q3 * q3 + q4 * q4 + q5 * q5)


def trace_Q3(q1, q2, q3, q4, q5):
    return 3 * (2 * q3 * (q3 * q3 - q1 * q1) + 2 * q2 * q4 * q5
                - (q1 - q3) * q5 * q5 + (q1 + q3) * q4 * q4 - 2 * q3 * q2 * q2)


def biaxiality(q1, q2, q3, q4, q5):
    """tr(Q^3)^2/tr(Q^2)^3, which is between 0 and 1/6"""
    return 1 - 6 * trace_Q3(q1, q2, q3, q4, q5) ** 2 / (trace_Q2(q1, q2, q3, q4, q5) ** 3 + 1e-14)


def Q_eigval(q1, q2, q3, q4, q5, largest=False):
    # coefficients
    p = -(q1 ** 2 + q2 ** 2 + 3 * q3 ** 2 + q4 ** 2 + q5 ** 2)
    q = -trace_Q3(q1, q2, q3, q4, q5) / 3
    return solve_cubic(np.zeros_like(q1), p, q, largest)


class LCFunc_s:
    def __init__(self, **kwargs):
        # Coefficients of LdG bulk
        A = kwargs.get("temp", -3900)
        B, C = kwargs.get("Landau_coeff", (6400, 3500))
        # Coefficient of LdG one-C elastic
        L = kwargs.get("elastic_coeff", 4e-11)

        # Characteristic length (nano-scale)
        lam = kwargs.get("char_len", 1e-6)
        # volume
        v0 = kwargs.get("volume", 0.09)
        # An artificial constant
        alpha = kwargs.get("alpha", .03125)  # alpha=1/32 then beta=1/2
        # Penalty factors
        wv = kwargs.get("w_void", 10)  # For void
        wa = kwargs.get("w_anch", 5)  # For anchoring
        wp = kwargs.get("w_phi", 1)  # For |∇φ|^2 (part of mixing energy)
        self.reset_params(A, B, C, L, lam, v0,
                          alpha, wv, wp, wa)
        self.aux = None

    def reset_params(self, A, B, C, L, lam, v0,
                     alpha, wv, wp, wa,
                     show=False):
        if not 0. < 12 * alpha < 1.:
            raise ValueError("α out of range. Should be in (0, 1/12).")
        # Physical constants
        self.A = A
        self.B = B
        self.C = C
        self.L = L
        self.lam = lam
        self.v0 = v0
        # Compute nondimensional coefficients
        barlam = lam * np.sqrt(C / L)
        sp = (B + np.sqrt(B * B - 24 * A * C)) / 4 / C  # s_plus
        bmin = sp ** 2 * (9 * A - B * sp) / 54  # minimum of bulk
        w2 = -bmin / alpha / C
        beta = 4 * (alpha + np.sqrt(alpha * (alpha + .25)))
        wv = max(wv, (B ** 2 / 27 - A * C) / ((C * (1 - beta)) ** 2))  # to induce strict minima

        # Display the functional
        if show:
            print("Full energy functional:\n")
            print("   ⎧")
            print("   ⎪   λ [ ")
            print("   ⎭ Ω       A·|Q|²/2C - B·tr(Q³)/3C + |Q|⁴/4")
            print("           + w₁·(1-φ)²·|Q|²/2")
            print("           + w₂·(1-φ)²·(φ²/4-2αφ-α)")
            print("         ] dx")
            print("   ⎧")
            print(" + ⎪   λ⁻¹ [")
            print("   ⎭ Ω         |∇Q|²/2 + w₃·|∇φ|²/2")
            print("             + w₄·|(Q+s₊/3)∇φ|²")
            print("           ] dx")
            print("with\tλ  = %.3e" % barlam)
            print("\tw₁ = %.3e\n\tw₂ = %.3e\n\tw₃ = %.3e\n\tw₄ = %.3e" % (wv, w2, wp, wa))
            print("\tα  = %.3e" % (alpha))

        # Store parameters into fields
        self.sp = sp
        self.bmin = bmin / C  # minimum of \bar{F}_b
        self.we = 1 / barlam  # elastic energy 1/2*|∇Q|^2
        self.wb = barlam  # bulk energy \bar{F}_b
        self.wb2 = A / C  # tr(Q^2)/2 in bulk
        self.wb3 = -B / C  # tr(Q^3)/3 in bulk
        self.wv = wv  # λ*1/2*(1-φ)^2|Q^2| in void
        self.wa = wa  # 1/λ*|(...)∇φ|^2 in anchoring
        self.wp1 = wp  # 1/λ*1/2*|∇φ|^2
        self.wp0 = w2  # λ*g(φ) in overall bulk energy
        self.alpha = alpha

    def reset_conf(self, conf: LCConfig, alpha=None, show=False):
        if alpha is None:
            alpha = self.alpha  # Do not change alpha
        self.reset_params(conf['A'], conf['B'], conf['C'], conf['L'],
                          conf['lam'], conf['vol'],
                          conf['alpha'], conf['wv'], conf['wp'], conf['wa'],
                          show)

    def export_conf(self, fn):
        save_lc_config(fn, LCConfig(A=self.A, B=self.B, C=self.C, L=self.L,
                                    lam=self.lam, vol=self.v0,
                                    alpha=self.alpha,
                                    wv=self.wv, wa=self.wa, wp=self.wp1))

    class LCAux:
        """Auxiliary variables that are used repeatedly"""

        def __init__(self, N: int):
            self.N = N
            one2n = np.arange(1, N + 1)
            k1, k2, k3 = np.meshgrid(one2n, one2n, one2n, indexing='ij')
            # grid of indices
            self.k1 = k1
            self.k2 = k2
            self.k3 = k3
            # N*N zero padding
            self.pad = np.zeros(N * N)
            # gradient square coefficients
            self.c_lap = (.125 * np.pi ** 2) * (k1 ** 2 + k2 ** 2 + k3 ** 2)
            # int of sin(kπx): 2/kπ if k odd else 0
            ios = 2. / np.pi / one2n
            ios[1::2] = 0.
            # # 1D collocation weights (manual scaling)
            # w_1d=1./(N+1)*fft.dst(ios,type=1)
            # self.w_col=w_1d[k1-1]*w_1d[k2-1]*w_1d[k3-1]
            # integral of sines in 3d
            self.ios3 = ios[k1 - 1] * ios[k2 - 1] * ios[k3 - 1]
            self.ios3_norm2 = norm(ios) ** 6

    def get_aux(self, N: int):
        self.aux = LCFunc_s.LCAux(N)

    def project(self, x: LCState_s, v0):
        """Project phi to `v0`"""
        if self.aux is None or self.aux.N != x.N:
            self.aux = LCFunc_s.LCAux(x.N)
        ios3 = self.aux.ios3
        r = self.aux.ios3_norm2
        x.phi -= (np.sum(ios3 * x.phi) - v0) * ios3 / r

    def project_vec(self, x: np.ndarray, N, v0):
        """Project phi to v0, but with vector input"""
        if self.aux is None or self.aux.N != N:
            self.aux = LCFunc_s.LCAux(N)
        ios3 = self.aux.ios3.ravel()
        r = self.aux.ios3_norm2
        xp = np.copy(x)
        xp[5 * N ** 3:] -= (np.dot(x[5 * N ** 3:], ios3) - v0) * ios3 / r
        return xp

    def padded_dct_and_dst(self, x, axis):
        """DCT in one axis and DST in the rest
        with zero padded before and after the input"""
        N, n2, n3 = x.shape
        assert N == n2 and n2 == n3
        if self.aux is None or self.aux.N != N:
            self.aux = LCFunc_s.LCAux(N)
        if axis == 0:
            return .125 * fft.dstn(fft.dctn(
                np.concatenate([self.aux.pad.reshape([1, N, N]), x], axis=0),
                type=1, s=N + 2, axes=0)[1:N + 1, :, :],
                                   type=1, axes=(1, 2))
        elif axis == 1:
            return .125 * fft.dstn(fft.dctn(
                np.concatenate([self.aux.pad.reshape([N, 1, N]), x], axis=1),
                type=1, s=N + 2, axes=1)[:, 1:N + 1, :],
                                   type=1, axes=(0, 2))
        elif axis == 2:
            return .125 * fft.dstn(fft.dctn(
                np.concatenate([self.aux.pad.reshape([N, N, 1]), x], axis=2),
                type=1, s=N + 2, axes=2)[:, :, 1:N + 1],
                                   type=1, axes=(0, 1))

    def energy(self, x: LCState_s):
        N = x.N
        h3 = 1. / (N + 1) ** 3
        x_v = x.sine_trans()
        # q1, q2, q3, q4, q5, phi = x.values()
        if self.aux is None:
            self.aux = LCFunc_s.LCAux(N)
        E = 0.
        # elastic energy
        E += self.we * \
             np.sum(self.aux.c_lap * (x.q1 ** 2 + x.q2 ** 2 + 3 * x.q3 ** 2
                                      + x.q4 ** 2 + x.q5 ** 2))
        # bulk energy
        trQ2 = trace_Q2(x_v.q1, x_v.q2, x_v.q3, x_v.q4, x_v.q5)
        trQ3 = trace_Q3(x_v.q1, x_v.q2, x_v.q3, x_v.q4, x_v.q5)
        bulk = self.wb2 / 2 * trQ2 + self.wb3 / 3 * trQ3 + .25 * trQ2 ** 2
        E += self.wb * (h3 * np.sum(bulk) - self.bmin)
        # mixing energy laplacian
        E += self.we * self.wp1 * .5 * \
             np.sum(self.aux.c_lap * x.phi ** 2)
        # mixing energy double-well (MODIFIED!!!)
        E += self.wb * self.wp0 * h3 * \
             np.sum((1. - x_v.phi) ** 2
                    * (.25 * x_v.phi ** 2 - self.alpha * (2 * x_v.phi + 1)))
        # anchoring energy
        phix = self.padded_dct_and_dst(self.aux.k1 * np.pi * x.phi, axis=0)
        phiy = self.padded_dct_and_dst(self.aux.k2 * np.pi * x.phi, axis=1)
        phiz = self.padded_dct_and_dst(self.aux.k3 * np.pi * x.phi, axis=2)
        x1, x2, x3 = Qtimes(x_v.q1, x_v.q2, x_v.q3, x_v.q4, x_v.q5, phix, phiy, phiz)
        anch_0 = x1 ** 2 + x2 ** 2 + x3 ** 2 + self.sp * 2 / 3 * (x1 * phix + x2 * phiy + x3 * phiz)
        E += self.we * self.wa * (h3 * np.sum(anch_0) + self.sp ** 2 / 9 * np.sum(self.aux.c_lap * x.phi ** 2))
        # void energy
        v = (1 - x_v.phi) ** 2 * trQ2
        E += self.wb * self.wv * .5 * h3 * np.sum(v)
        return E

    def energy_vec(self, x: np.ndarray, N, proj=True):
        """To prevent roundoff error, project x to v0 when input is vector"""
        if proj:
            x = self.project_vec(x, N, self.v0)
        return self.energy(view_as_lc(x, N))

    def grad(self, x: LCState_s, nonlin_only=False):
        """Gradient of functional
        `nonlin_only`: whether to evaluate only the nonlinear terms (i.e. excluding the
                       Laplacians)"""
        N = x.N
        h3 = 1. / (N + 1) ** 3
        # q1, q2, q3, q4, q5, phi = x.values()
        x_v = x.sine_trans()
        q1, q2, q3, q4, q5, phi = x_v.q1, x_v.q2, x_v.q3, x_v.q4, x_v.q5, x_v.phi
        g = LCState_s(N)  # gradient
        g_v = LCState_s(N)  # gradient w.r.t. values
        if self.aux is None:
            self.aux = LCFunc_s.LCAux(N)

        # LdG elastic
        if not nonlin_only:
            g.q1[:] = self.we * 2 * x.q1 * self.aux.c_lap
            g.q2[:] = self.we * 2 * x.q2 * self.aux.c_lap
            g.q3[:] = self.we * 6 * x.q3 * self.aux.c_lap
            g.q4[:] = self.we * 2 * x.q4 * self.aux.c_lap
            g.q5[:] = self.we * 2 * x.q5 * self.aux.c_lap
        # LdG bulk
        trQ2 = trace_Q2(q1, q2, q3, q4, q5)
        QpA_C = trQ2 + self.wb2
        g_v.q1[:] = self.wb * h3 * \
                    (2 * QpA_C * q1 + self.wb3 * (-4 * q3 * q1 - q5 * q5 + q4 * q4))
        g_v.q3[:] = self.wb * h3 * \
                    (6 * QpA_C * q3 + self.wb3 * (6 * q3 * q3 - 2 * q1 * q1
                                                  + q5 * q5 + q4 * q4 - 2 * q2 * q2))
        g_v.q2[:] = self.wb * h3 * \
                    (2 * QpA_C * q2 + self.wb3 * (2 * q4 * q5 - 4 * q3 * q2))
        g_v.q4[:] = self.wb * h3 * \
                    (2 * QpA_C * q4 + self.wb3 * (2 * q2 * q5 + 2 * (q1 + q3) * q4))
        g_v.q5[:] = self.wb * h3 * \
                    (2 * QpA_C * q5 + self.wb3 * (2 * q2 * q4 - 2 * (q1 - q3) * q5))
        # mixing laplacian
        if not nonlin_only:
            g.phi[:] = self.we * self.wp1 * self.aux.c_lap * x.phi
        # mixing double-well (MODIFIED)
        # use the derivative g'(x) = x(x-1)(x-.5-6α)
        g_v.phi[:] = self.wb * self.wp0 * h3 * \
                     phi * (phi - 1) * (phi - (.5 + 6 * self.alpha))
        # anchoring
        phix = self.padded_dct_and_dst(self.aux.k1 * np.pi * x.phi, axis=0)
        phiy = self.padded_dct_and_dst(self.aux.k2 * np.pi * x.phi, axis=1)
        phiz = self.padded_dct_and_dst(self.aux.k3 * np.pi * x.phi, axis=2)
        x1, x2, x3 = Qtimes(q1, q2, q3, q4, q5, phix, phiy, phiz)
        g_v.q1[:] += h3 * self.we * self.wa * \
                     (2 * (x1 * phix - x2 * phiy)
                      + self.sp * 2 / 3 * (phix ** 2 - phiy ** 2))
        g_v.q3[:] += h3 * self.we * self.wa * \
                     (2 * (-x1 * phix - x2 * phiy + 2 * x3 * phiz)
                      + self.sp * 2 / 3 * (-phix ** 2 - phiy ** 2 + 2 * phiz ** 2))
        g_v.q2[:] += h3 * self.we * self.wa * \
                     (2 * (x1 * phiy + x2 * phix) + self.sp * 4 / 3 * phix * phiy)
        g_v.q4[:] += h3 * self.we * self.wa * \
                     (2 * (x1 * phiz + x3 * phix) + self.sp * 4 / 3 * phix * phiz)
        g_v.q5[:] += h3 * self.we * self.wa * \
                     (2 * (x2 * phiz + x3 * phiy) + self.sp * 4 / 3 * phiy * phiz)
        d_a0_d_phix, d_a0_d_phiy, d_a0_d_phiz = Qtimes(q1, q2, q3, q4, q5,
                                                       2 * x1 + self.sp * 4 / 3 * phix,
                                                       2 * x2 + self.sp * 4 / 3 * phiy,
                                                       2 * x3 + self.sp * 4 / 3 * phiz)
        g.phi[:] += self.we * self.wa * \
                    (h3 * np.pi * (self.aux.k1 * self.padded_dct_and_dst(d_a0_d_phix, axis=0)
                                   + self.aux.k2 * self.padded_dct_and_dst(d_a0_d_phiy, axis=1)
                                   + self.aux.k3 * self.padded_dct_and_dst(d_a0_d_phiz, axis=2))
                     + self.sp ** 2 * 2 / 9 * self.aux.c_lap * x.phi)
        # void
        g_v.q1[:] += self.wb * self.wv * h3 * 2 * (1 - phi) ** 2 * q1
        g_v.q2[:] += self.wb * self.wv * h3 * 2 * (1 - phi) ** 2 * q2
        g_v.q3[:] += self.wb * self.wv * h3 * 6 * (1 - phi) ** 2 * q3
        g_v.q4[:] += self.wb * self.wv * h3 * 2 * (1 - phi) ** 2 * q4
        g_v.q5[:] += self.wb * self.wv * h3 * 2 * (1 - phi) ** 2 * q5
        g_v.phi[:] += self.wb * self.wv * h3 * (phi - 1) * trQ2

        # chain rule: apply (adjoint) DST to value derivatives
        g1 = g_v.sine_trans()
        g.x += g1.x
        return g

    def grad_vec(self, x: np.ndarray, N, proj=True):
        """Evaluate gradient on vector input
        To prevent roundoff errors, always project input"""
        if proj:
            x = self.project_vec(x, N, self.v0)
        g = self.grad(view_as_lc(x, N))
        if proj:
            g.proj_phi(0)
        return g.x

    def gradQ(self, x: LCState_s):
        """Only the Q-gradient"""
        g = self.grad(x)
        g.phi[:] = 0.  # clear the phi gradient
        return g

    def gradQ_vec(self, x: np.ndarray, N):
        return self.gradQ(view_as_lc(x, N)).x

    def hess(self, x: LCState_s):
        """Compute Hessian as implicit matrix (LinearOperator instance)
        Computes first variation of gradient from variation in x"""

        def _matvec(v):
            nonlocal self, x, N
            x_v = x.sine_trans()  # values
            dx = view_as_lc(v, N)  # variation
            dx_v = dx.sine_trans()  # variation values
            h3 = 1. / (N + 1) ** 3
            dg = LCState_s(N)  # gradient variation
            dg_v = LCState_s(N)  # gradient variation w.r.t. values

            # LdG elastic (linear Laplacian)
            dg.q1[:] = self.we * 2 * dx.q1 * self.aux.c_lap
            dg.q2[:] = self.we * 2 * dx.q2 * self.aux.c_lap
            dg.q3[:] = self.we * 6 * dx.q3 * self.aux.c_lap
            dg.q4[:] = self.we * 2 * dx.q4 * self.aux.c_lap
            dg.q5[:] = self.we * 2 * dx.q5 * self.aux.c_lap
            # LdG bulk
            trQ2 = trace_Q2(x_v.q1, x_v.q2, x_v.q3, x_v.q4, x_v.q5)
            dtrQ2 = 4 * (x_v.q1 * dx_v.q1
                         + x_v.q2 * dx_v.q2
                         + 3 * x_v.q3 * dx_v.q3
                         + x_v.q4 * dx_v.q4
                         + x_v.q5 * dx_v.q5)
            QpA_C = trQ2 + self.wb2
            dg_v.q1[:] = 2 * self.wb * h3 * \
                         (QpA_C * dx_v.q1 + dtrQ2 * x_v.q1
                          + self.wb3 * (-2 * (x_v.q3 * dx_v.q1 + dx_v.q3 * x_v.q1)
                                        - x_v.q5 * dx_v.q5 + x_v.q4 * dx_v.q4))
            dg_v.q3[:] = 2 * self.wb * h3 * \
                         (3 * (QpA_C * dx_v.q3 + dtrQ2 * x_v.q3)
                          + self.wb3 * (6 * x_v.q3 * dx_v.q3 - 2 * x_v.q1 * dx_v.q1
                                        + x_v.q5 * dx_v.q5 + x_v.q4 * dx_v.q4
                                        - 2 * x_v.q2 * dx_v.q2))
            dg_v.q2[:] = 2 * self.wb * h3 * \
                         (QpA_C * dx_v.q2 + dtrQ2 * x_v.q2
                          + self.wb3 * ((x_v.q4 * dx_v.q5 + x_v.q5 * dx_v.q4)
                                        - 2 * (x_v.q3 * dx_v.q2 + dx_v.q3 * x_v.q2)))
            dg_v.q4[:] = 2 * self.wb * h3 * \
                         (QpA_C * dx_v.q4 + dtrQ2 * x_v.q4
                          + self.wb3 * (x_v.q2 * dx_v.q5 + x_v.q5 * dx_v.q2
                                        + ((x_v.q1 + x_v.q3) * dx_v.q4
                                           + (dx_v.q1 + dx_v.q3) * x_v.q4)))
            dg_v.q5[:] = 2 * self.wb * h3 * \
                         (QpA_C * dx_v.q5 + dtrQ2 * x_v.q5
                          + self.wb3 * (x_v.q2 * dx_v.q4 + x_v.q4 * dx_v.q2
                                        - ((x_v.q1 - x_v.q3) * dx_v.q5
                                           + (dx_v.q1 - dx_v.q3) * x_v.q5)))
            # mixing laplacian
            dg.phi[:] = self.we * self.wp1 * self.aux.c_lap * dx.phi
            # mixing double-well (MODIFIED)
            # use the derivative g'(x) = x(x-1)(x-.5-6α)
            dg_v.phi[:] = self.wb * self.wp0 * h3 * \
                          ((2 * x_v.phi - 1) * (x_v.phi - (.5 + 6 * self.alpha))
                           + x_v.phi * (x_v.phi - 1)) * dx_v.phi
            # anchoring
            phix = self.padded_dct_and_dst(self.aux.k1 * np.pi * x.phi, axis=0)
            phiy = self.padded_dct_and_dst(self.aux.k2 * np.pi * x.phi, axis=1)
            phiz = self.padded_dct_and_dst(self.aux.k3 * np.pi * x.phi, axis=2)
            dphix = self.padded_dct_and_dst(self.aux.k1 * np.pi * dx.phi, axis=0)
            dphiy = self.padded_dct_and_dst(self.aux.k2 * np.pi * dx.phi, axis=1)
            dphiz = self.padded_dct_and_dst(self.aux.k3 * np.pi * dx.phi, axis=2)
            x1, x2, x3 = Qtimes(x_v.q1, x_v.q2, x_v.q3, x_v.q4, x_v.q5,
                                phix, phiy, phiz)
            dx1, dx2, dx3 = dQtimesv(x_v.q1, x_v.q2, x_v.q3, x_v.q4, x_v.q5,
                                     phix, phiy, phiz,
                                     dx_v.q1, dx_v.q2, dx_v.q3, dx_v.q4, dx_v.q5,
                                     dphix, dphiy, dphiz)
            dg_v.q1[:] += h3 * self.we * self.wa * \
                          (2 * (dx1 * phix + x1 * dphix - dx2 * phiy - x2 * dphiy)
                           + self.sp * 4 / 3 * (phix * dphix - phiy * dphiy))
            dg_v.q3[:] += h3 * self.we * self.wa * \
                          (2 * (-dx1 * phix - x1 * dphix - dx2 * phiy - x2 * dphiy
                                + 2 * (dx3 * phiz + x3 * dphiz))
                           + self.sp * 4 / 3 * (-phix * dphix - phiy * dphiy + 2 * phiz * dphiz))
            dg_v.q2[:] += h3 * self.we * self.wa * \
                          (2 * (dx1 * phiy + x1 * dphiy + dx2 * phix + x2 * dphix)
                           + self.sp * 4 / 3 * (dphix * phiy + phix * dphiy))
            dg_v.q4[:] += h3 * self.we * self.wa * \
                          (2 * (dx1 * phiz + x1 * dphiz + dx3 * phix + x3 * dphix)
                           + self.sp * 4 / 3 * (dphix * phiz + phix * dphiz))
            dg_v.q5[:] += h3 * self.we * self.wa * \
                          (2 * (dx2 * phiz + x2 * dphiz + dx3 * phiy + x3 * dphiy)
                           + self.sp * 4 / 3 * (dphiy * phiz + phiy * dphiz))
            dd_a0_d_phix, dd_a0_d_phiy, dd_a0_d_phiz = dQtimesv(
                x_v.q1, x_v.q2, x_v.q3, x_v.q4, x_v.q5,
                2 * x1 + self.sp * 4 / 3 * phix,
                2 * x2 + self.sp * 4 / 3 * phiy,
                2 * x3 + self.sp * 4 / 3 * phiz,
                dx_v.q1, dx_v.q2, dx_v.q3, dx_v.q4, dx_v.q5,
                2 * dx1 + self.sp * 4 / 3 * dphix,
                2 * dx2 + self.sp * 4 / 3 * dphiy,
                2 * dx3 + self.sp * 4 / 3 * dphiz)
            dg.phi[:] += self.we * self.wa * \
                         (h3 * np.pi * (self.aux.k1 * self.padded_dct_and_dst(dd_a0_d_phix, axis=0)
                                        + self.aux.k2 * self.padded_dct_and_dst(dd_a0_d_phiy, axis=1)
                                        + self.aux.k3 * self.padded_dct_and_dst(dd_a0_d_phiz, axis=2))
                          + self.sp ** 2 * 2 / 9 * self.aux.c_lap * dx.phi)
            # void
            _1mphi2 = (1 - x_v.phi) ** 2
            d1mphi2 = -2 * (1 - x_v.phi) * dx_v.phi
            dg_v.q1[:] += self.wb * self.wv * h3 * \
                          2 * (d1mphi2 * x_v.q1 + _1mphi2 * dx_v.q1)
            dg_v.q2[:] += self.wb * self.wv * h3 * \
                          2 * (d1mphi2 * x_v.q2 + _1mphi2 * dx_v.q2)
            dg_v.q3[:] += self.wb * self.wv * h3 * \
                          6 * (d1mphi2 * x_v.q3 + _1mphi2 * dx_v.q3)
            dg_v.q4[:] += self.wb * self.wv * h3 * \
                          2 * (d1mphi2 * x_v.q4 + _1mphi2 * dx_v.q4)
            dg_v.q5[:] += self.wb * self.wv * h3 * \
                          2 * (d1mphi2 * x_v.q5 + _1mphi2 * dx_v.q5)
            dg_v.phi[:] += self.wb * self.wv * h3 * \
                           (dx_v.phi * trQ2 + (x_v.phi - 1) * dtrQ2)

            # chain rule: apply (adjoint) DST to value derivatives
            dg1 = dg_v.sine_trans()
            dg.x += dg1.x
            # Project gradient to meet volume constraint
            # self.project(g, 0)
            return dg.x

        N = x.N
        if self.aux is None:
            self.aux = LCFunc_s.LCAux(N)
        return LinearOperator(dtype=float, shape=(6 * N ** 3, 6 * N ** 3),
                              matvec=_matvec)

    def grad_descent(self,
                     X0: LCState_s,
                     maxiter=10000,
                     eta=1e-8,
                     tol=1e-8,
                     bb=False,
                     verbose=0,
                     inspect=False):
        X = LCState_s(X0.N)
        X.x[:] = X0.x
        if inspect:
            fvec = np.zeros(maxiter)
        flag = 0
        Gp = np.zeros_like(X.x)
        s = np.zeros_like(X.x)
        for k in range(maxiter):
            G = self.grad(X)
            G.proj_phi(0)
            if verbose >= 2:
                print("Gradient norm @ Iter %d: %.3e" % (k, norm(G)))
            if (gnorm := norm(G.x)) < tol:
                if verbose:
                    print("Iter over @ No.", k, ", |g| =", gnorm)
                flag = 1
                break
            a = eta
            if bb and k > 0:
                y = G.x - Gp
                a = np.abs(np.dot(s, y) / np.dot(y, y))
                # if a < -1e-2:
                #     a = eta  # Keep step length positive
            # print(k,"::", a, gnorm)
            # Descent
            X.x[:] -= a * G.x
            if np.any(np.isnan(X.x)) and verbose:
                print("NAN at step", k)
                flag = -1
                break
            if bb:
                s[:] = -a * G.x
                Gp[:] = G.x
            if inspect:
                fvec[k] = self.energy(X)
        if flag == 0 and verbose:
            print("Iteration failed to converge, |g| =", gnorm)
        if inspect:
            return X, flag, fvec[0:k]
        return X, flag


def test_grad(FF: LCFunc_s, X: LCState_s, n=5):
    for _ in range(n):
        v = np.random.randn(len(X.x))
        for eps in 1e-3, 1e-4, 1e-5, 1e-6, 1e-7:
            dfdv = (FF.energy_vec(X.x + eps * v, X.N)
                    - FF.energy_vec(X.x - eps * v, X.N)) / 2 / eps
            gdotv = np.dot(FF.grad_vec(X.x, X.N), v)
            print(np.log10(norm(dfdv - gdotv)), end=' ')
        print()


def test_hess(FF: LCFunc_s, X: LCState_s, n=5):
    H = FF.hess(X)
    for _ in range(n):
        v = np.random.randn(len(X.x))
        for eps in 1e-3, 1e-4, 1e-5, 1e-6, 1e-7:
            dgdv = (FF.grad_vec(X.x + eps * v, X.N, proj=False)
                    - FF.grad_vec(X.x - eps * v, X.N, proj=False)) / 2 / eps
            Hv = H @ v
            print(np.log10(norm(dgdv - Hv)), end=' ')
        print()


def test_bulk(FF: LCFunc_s, n=10):
    bmin = FF.sp ** 2 * (9 * FF.A - FF.B * FF.sp) / (54 * FF.C)  # minimum of bulk (rescaled)
    for _ in range(n):
        q1, q2, q3, q4, q5, phi = np.random.randn(6)
        trQ2 = trace_Q2(q1, q2, q3, q4, q5)
        trQ3 = trace_Q3(q1, q2, q3, q4, q5)
        W = (FF.wb2 / 2 * trQ2 + FF.wb3 / 3 * trQ3 + .25 * trQ2 ** 2) \
            + FF.wv / 2 * (1 - phi) ** 2 * trQ2 \
            + FF.wp0 * (1 - phi) ** 2 * (phi ** 2 / 4 - FF.alpha * (2 * phi + 1))
        if W < bmin:
            print("Bulk is wrong! W=%f, bmin=%f" % (W, bmin))
            return False
    return True


if __name__ == "__main__":
    N = 15
    FF = LCFunc_s()
    X = LCState_s(N)
    # Random, but with regularization
    np.random.seed(142)
    FF.get_aux(N)
    reg = np.exp(-(FF.aux.k1 ** 2 + FF.aux.k2 ** 2 + FF.aux.k3 ** 2) / 2)
    X.q1[:] = np.random.randn(N, N, N) * reg
    X.q2[:] = np.random.randn(N, N, N) * reg
    X.q3[:] = np.random.randn(N, N, N) * reg
    X.q4[:] = np.random.randn(N, N, N) * reg
    X.q5[:] = np.random.randn(N, N, N) * reg
    X.phi[:] = np.random.randn(N, N, N) * reg
    test_grad(FF, X)
    test_hess(FF, X)
    print(test_bulk(FF, 10))
