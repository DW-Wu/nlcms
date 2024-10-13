"""Computations of the LC state"""
from typing import Any
from nlc_state import *
import json
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
    p = (3 * b - a**2) / 3
    q = (9 * a * b - 27 * c - 2 * a**3) / 27
    if largest:
        x = np.maximum(1, np.sqrt(np.abs(p) + np.abs(q))) - a / 3  # x0 > largest root
    else:
        x = -np.maximum(1, np.sqrt(np.abs(p) + np.abs(q))) - a / 3  # x0 < smallest root
    for _ in range(maxiter):
        f = x**3 + a * x**2 + b * x + c
        if np.max(np.abs(f)) < 1e-10:
            # print('Newton successful')
            break
        x -= (x**3 + a * x**2 + b * x + c) / (3 * x**2 + 2 * a * x + b)
    return x


def trace_Q2(q1, q2, q3, q4, q5):
    return 2 * (q1 * q1 + q2 * q2 + 3 * q3 * q3 + q4 * q4 + q5 * q5)


def trace_Q3(q1, q2, q3, q4, q5):
    return 3 * (2 * q3 * (q3 * q3 - q1 * q1) + 2 * q2 * q4 * q5
                - (q1 - q3) * q5 * q5 + (q1 + q3) * q4 * q4 - 2 * q3 * q2 * q2)


def biaxiality(q1, q2, q3, q4, q5):
    """tr(Q^3)^2/tr(Q^2)^3, which is between 0 and 1/6"""
    return 1 - 6 * trace_Q3(q1, q2, q3, q4, q5)**2 / (trace_Q2(q1, q2, q3, q4, q5)**3 + 1e-14)


def Q_eigval(q1, q2, q3, q4, q5, largest=False):
    # coefficients
    p = -(q1**2 + q2**2 + 3 * q3**2 + q4**2 + q5**2)
    q = -trace_Q3(q1, q2, q3, q4, q5) / 3
    return solve_cubic(np.zeros_like(q1), p, q, largest)


class LCConfig(dict):
    """LdG+PhF model configuration object. Same as python dictionary"""

    def __init__(self, **kwargs):
        p_default = {'A': -3900, 'B': 6400, 'C': 3500, 'L': 4e-11,
                     'lam': 1e-6, 'v0': 0.04,
                     'eps': .01, 'omega': 5, 'wp': 1, 'wv': 1}
        p_default.update(kwargs)
        super().__init__(**p_default)
        for key, value in p_default.items():
            setattr(self, key, value)

    def __setattr__(self, __name: str, __value):
        if __name not in self.keys():
            raise KeyError("Invalid key %s for LCConfig; key must be in %a" % (__name,
                                                                               list(self.keys())))
        super().__setitem__(__name, __value)  # change dict values as well
        return super().__setattr__(__name, __value)


def save_lc_config(fn, x):
    """Save configuration as json"""
    if not fn.endswith('.json'):
        fn += '.json'
    with open(fn, 'w') as f:
        f.writelines(json.dumps(dict(x), separators=(',', ': '), indent=2))


def load_lc_config(fn):
    """Load configuration from json"""
    with open(fn, 'rb') as f:
        D = dict(json.loads(f.read()))
    return LCConfig(**D)


class LCFunc_s:
    def __init__(self, **kwargs):
        self.reset_params(A=kwargs.get("temp", -3900),
                          B=kwargs.get("bulk3", 6400),
                          C=kwargs.get("bulk4", 3500),
                          L=kwargs.get("elastic", 4e-11),
                          lam=kwargs.get("char_len", 1e-6),
                          v0=kwargs.get("volume", 0.09),
                          eps=kwargs.get("capillary_width", 0.01),
                          omega=kwargs.get("w_anch", 1),
                          wp=kwargs.get("w_mix", 5),
                          wv=kwargs.get("w_void", 1))
        # Penalty factors
        self.aux = None

    def reset_params(self, show=False, A=-3900., B=6400., C=3500., L=4e-11,
                     lam=1e-6, v0=.04, eps=.01,
                     omega=5, wp=1, wv=1):
        # Compute nondimensional coefficients
        barlam = lam * np.sqrt(C / L)
        sp = (B + np.sqrt(B * B - 24 * A * C)) / 4 / C  # s_plus
        bmin = sp**2 * (9 * A - B * sp) / 54  # minimum of bulk

        # Display the functional
        if show:
            print("Full energy functional:\n")
            print("   ⎧")
            print("   ⎪   φ²[ λ·(A·|Q|²/2C - B·tr(Q³)/3C + |Q|⁴/4) + λ⁻¹·|∇Q|²/2 ] dx")
            print("   ⎭ Ω")
            print("")
            print("      ⎧")
            print(" + w₁ ⎪   [ ε⁻¹·(1-φ)²·φ² + ε·(|∇φ|² + ω·|(Q+s₊/3)∇φ|²) ] dx")
            print("      ⎭ Ω")
            print("")
            print("      ⎧")
            print(" + w₂ ⎪   [ κ⁻¹·(1-φ)²·|Q|²/2 + κ·|∇Q|²/2 ] dx")
            print("      ⎭ Ω")
            print("with\tλ  = %.3e" % barlam)
            print("\tw₁ = %.3e\n\tw₂ = %.3e\n\tε = %.3e\n\tω = %.3e\n\tκ = √(ε) = %.3e"
                  % (wp, wv, eps, omega, np.sqrt(eps)))

        # Compute coefficients
        self.conf = LCConfig(A=A, B=B, C=C, L=L,
                             lam=lam, v0=v0, eps=eps,
                             omega=omega, wp=wp, wv=wv)
        self.sp = sp
        self.v0 = v0
        self.bmin = bmin / C  # minimum of \bar{F}_b
        self.we = 1 / barlam  # elastic energy 1/2*φ^2*|∇Q|^2
        self.wb = barlam  # bulk energy \bar{F}_b
        self.wb2 = A / C  # tr(Q^2)/2 in bulk
        self.wb3 = -B / C  # tr(Q^3)/3 in bulk
        self.wv = wv / np.sqrt(eps)  # 1/2*(1-φ)^2|Q^2| in void
        self.wv1 = wv * np.sqrt(eps)  # an extra 1/2*|∇Q|^2 term
        self.wa = wp * eps * omega  # |(...)∇φ|^2 in anchoring
        self.wp1 = 2 * wp * eps  # 1/2*|∇φ|^2 in mixing
        self.wp0 = wp / eps  # (1-φ)^2*φ^2 in mixing

    def reset_conf(self, conf: LCConfig, show=False):
        self.conf.update(conf)
        self.reset_params(show, **conf)

    def export_conf(self, fn):
        save_lc_config(fn, self.conf)

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
            self.c_lap = (.125 * np.pi**2) * (k1**2 + k2**2 + k3**2)
            # int of sin(kπx): 2/kπ if k odd else 0
            ios = 2. / np.pi / one2n
            ios[1::2] = 0.
            # # 1D collocation weights (manual scaling)
            # w_1d=1./(N+1)*fft.dst(ios,type=1)
            # self.w_col=w_1d[k1-1]*w_1d[k2-1]*w_1d[k3-1]
            # integral of sines in 3d
            self.ios3 = ios[k1 - 1] * ios[k2 - 1] * ios[k3 - 1]
            self.ios3_norm2 = norm(ios)**6

    def get_aux(self, N: int):
        self.aux = LCFunc_s.LCAux(N)

    def volume(self, x: LCState_s):
        if self.aux is None or self.aux.N != x.N:
            self.aux = LCFunc_s.LCAux(x.N)
        return np.sum(self.aux.ios3 * x.phi)

    def project(self, x: LCState_s, v0):
        """Project phi to v0
        Modifies in-place"""
        if self.aux is None or self.aux.N != x.N:
            self.aux = LCFunc_s.LCAux(x.N)
        ios3 = self.aux.ios3
        r = self.aux.ios3_norm2
        x.phi -= (np.sum(ios3 * x.phi) - v0) * ios3 / r

    def project_vec(self, x: np.ndarray, N, v0):
        """Project phi to v0, but with vector input.
        Returns new vector"""
        if self.aux is None or self.aux.N != N:
            self.aux = LCFunc_s.LCAux(N)
        ios3 = self.aux.ios3.ravel()
        r = self.aux.ios3_norm2
        xp = np.copy(x)
        xp[5 * N**3:] -= (np.dot(x[5 * N**3:], ios3) - v0) * ios3 / r
        return xp

    def energy(self, x: LCState_s, part='all'):
        """Energy with LdG density masked by φ"""
        N = x.N
        h3 = 1. / (N + 1)**3
        xv = x.sine_trans()  # function values
        xvx = x.xdiff()  # derivatives
        xvy = x.ydiff()
        xvz = x.zdiff()
        q1, q2, q3, q4, q5, phi = \
            xv.q1, xv.q2, xv.q3, xv.q4, xv.q5, xv.phi
        if self.aux is None:
            self.aux = LCFunc_s.LCAux(N)
        trQ2 = trace_Q2(q1, q2, q3, q4, q5)
        E = 0.
        if part == 'e' or part == 'all':
            # elastic energy
            E += self.we * h3 * \
                 np.sum(phi**2 * (xvx.q1**2 + xvx.q2**2 + 3 * xvx.q3**2 + xvx.q4**2 + xvx.q5**2
                                  + xvy.q1**2 + xvy.q2**2 + 3 * xvy.q3**2 + xvy.q4**2 + xvy.q5**2
                                  + xvz.q1**2 + xvz.q2**2 + 3 * xvz.q3**2 + xvz.q4**2 + xvz.q5**2))
        if part == 'b' or part == 'all':
            # bulk energy
            trQ3 = trace_Q3(q1, q2, q3, q4, q5)
            bulk = self.wb2 / 2 * trQ2 + self.wb3 / 3 * trQ3 + .25 * trQ2**2 - self.bmin
            E += self.wb * h3 * np.sum(phi**2 * bulk)
        if part == 'm' or part == 'all':
            # mixing energy laplacian
            E += self.wp1 * .5 * np.sum(self.aux.c_lap * x.phi**2)
            # mixing energy double-well
            E += self.wp0 * h3 * np.sum(((1. - phi) * phi)**2)
        if part == 'a' or part == 'all':
            # anchoring energy
            x1, x2, x3 = Qtimes(q1, q2, q3, q4, q5, xvx.phi, xvy.phi, xvz.phi)
            anch_0 = x1**2 + x2**2 + x3**2 + \
                     2 / 3 * self.sp * (x1 * xvx.phi + x2 * xvy.phi + x3 * xvz.phi)
            E += self.wa * (h3 * np.sum(anch_0) + self.sp**2 / 9 * np.sum(self.aux.c_lap * x.phi**2))
        if part == 'v' or part == 'all':
            # void energy
            v = (1 - phi)**2 * trQ2
            E += .5 * self.wv * h3 * np.sum(v)
            # an extra gradient term for coercivity
            E += self.wv1 * np.sum(self.aux.c_lap * (x.q1**2
                                                     + x.q2**2
                                                     + 3 * x.q3**2
                                                     + x.q4**2
                                                     + x.q5**2))
        return E

    def energy_vec(self, x: np.ndarray, N, proj=True):
        """To prevent roundoff error, project x to v0 when input is vector"""
        if proj:
            x = self.project_vec(x, N, self.conf.v0)
        return self.energy(view_as_lc(x, N))

    def grad(self, x: LCState_s, part=2, proj=False):
        """Gradient of functional

        ** arguments **

        `part`: if part==0, evaluate only the linear terms (Laplacians)
                if part==1, evaluate only the nonlinear terms
                if part==2, evaluate the whole gradient"""
        N = x.N
        if self.aux is None:
            self.aux = LCFunc_s.LCAux(N)
        h3 = 1. / (N + 1)**3
        xv = x.sine_trans()  # function values
        xvx = x.xdiff(aux_k1=self.aux.k1)  # derivatives
        xvy = x.ydiff(aux_k2=self.aux.k2)
        xvz = x.zdiff(aux_k3=self.aux.k3)
        q1, q2, q3, q4, q5, phi = \
            xv.q1, xv.q2, xv.q3, xv.q4, xv.q5, xv.phi
        trQ2 = trace_Q2(q1, q2, q3, q4, q5)
        g = LCState_s(N)  # gradient
        g_v = LCState_s(N)  # gradient w.r.t. values

        # LdG elastic (now with mask, so evaluate with adjoints)
        if part != 1:
            for i in range(5):
                g.x4[i, :] = (2 if i != 2 else 6) * self.we * h3 * np.pi * (
                        self.aux.k1 * padded_dct_then_dst(phi**2 * xvx.x4[i, :], axis=0)
                        + self.aux.k2 * padded_dct_then_dst(phi**2 * xvy.x4[i, :], axis=1)
                        + self.aux.k3 * padded_dct_then_dst(phi**2 * xvz.x4[i, :], axis=2)
                )
        if part != 0:
            g_v.phi[:] = 2 * self.we * h3 * phi * (xvx.q1**2 + xvx.q2**2 + 3 * xvx.q3**2 + xvx.q4**2 + xvx.q5**2
                                                   + xvy.q1**2 + xvy.q2**2 + 3 * xvy.q3**2 + xvy.q4**2 + xvy.q5**2
                                                   + xvz.q1**2 + xvz.q2**2 + 3 * xvz.q3**2 + xvz.q4**2 + xvz.q5**2)
        # LdG bulk
        if part != 0:
            trQ3 = trace_Q3(q1, q2, q3, q4, q5)
            bulk = self.wb2 / 2 * trQ2 + self.wb3 / 3 * trQ3 + .25 * trQ2**2 - self.bmin
            QpA_C = trQ2 + self.wb2
            g_v.q1[:] = self.wb * h3 * phi**2 * \
                        (2 * QpA_C * q1 + self.wb3 * (-4 * q3 * q1 - q5 * q5 + q4 * q4))
            g_v.q3[:] = self.wb * h3 * phi**2 * \
                        (6 * QpA_C * q3 + self.wb3 * (6 * q3 * q3 - 2 * q1 * q1
                                                      + q5 * q5 + q4 * q4 - 2 * q2 * q2))
            g_v.q2[:] = self.wb * h3 * phi**2 * \
                        (2 * QpA_C * q2 + self.wb3 * (2 * q4 * q5 - 4 * q3 * q2))
            g_v.q4[:] = self.wb * h3 * phi**2 * \
                        (2 * QpA_C * q4 + self.wb3 * (2 * q2 * q5 + 2 * (q1 + q3) * q4))
            g_v.q5[:] = self.wb * h3 * phi**2 * \
                        (2 * QpA_C * q5 + self.wb3 * (2 * q2 * q4 - 2 * (q1 - q3) * q5))
            g_v.phi[:] += 2 * self.wb * h3 * (bulk * phi)
        # mixing energy
        if part != 0:
            g_v.phi[:] += 4 * self.wp0 * h3 * \
                          phi * (phi - 1) * (phi - .5)
        if part != 1:
            g.phi[:] = self.wp1 * self.aux.c_lap * x.phi
        # anchoring
        x1, x2, x3 = Qtimes(q1, q2, q3, q4, q5, xvx.phi, xvy.phi, xvz.phi)
        if part != 0:
            g_v.q1[:] += h3 * self.wa * \
                         (2 * (x1 * xvx.phi - x2 * xvy.phi)
                          + self.sp * 2 / 3 * (xvx.phi**2 - xvy.phi**2))
            g_v.q3[:] += h3 * self.wa * \
                         (2 * (-x1 * xvx.phi - x2 * xvy.phi + 2 * x3 * xvz.phi)
                          + self.sp * 2 / 3 * (-xvx.phi**2 - xvy.phi**2 + 2 * xvz.phi**2))
            g_v.q2[:] += h3 * self.wa * \
                         (2 * (x1 * xvy.phi + x2 * xvx.phi) + self.sp * 4 / 3 * xvx.phi * xvy.phi)
            g_v.q4[:] += h3 * self.wa * \
                         (2 * (x1 * xvz.phi + x3 * xvx.phi) + self.sp * 4 / 3 * xvx.phi * xvz.phi)
            g_v.q5[:] += h3 * self.wa * \
                         (2 * (x2 * xvz.phi + x3 * xvy.phi) + self.sp * 4 / 3 * xvy.phi * xvz.phi)
        if part != 1:
            # Diffusion of phi in anchoring energy
            d_a0_d_phix, d_a0_d_phiy, d_a0_d_phiz = \
                Qtimes(q1, q2, q3, q4, q5,
                       2 * x1 + self.sp * 4 / 3 * xvx.phi,
                       2 * x2 + self.sp * 4 / 3 * xvy.phi,
                       2 * x3 + self.sp * 4 / 3 * xvz.phi)
            g.phi[:] += self.wa * h3 * np.pi * \
                        (self.aux.k1 * padded_dct_then_dst(d_a0_d_phix, axis=0)
                         + self.aux.k2 * padded_dct_then_dst(d_a0_d_phiy, axis=1)
                         + self.aux.k3 * padded_dct_then_dst(d_a0_d_phiz, axis=2))
            g.phi[:] += self.wa * self.sp**2 * 2 / 9 * self.aux.c_lap * x.phi
        # void
        if part != 0:
            for i in range(5):
                g_v.x4[i, :] += (2 if i != 2 else 6) * self.wv * h3 * (1 - phi)**2 * xv.x4[i, :]
            g_v.phi[:] += self.wv * h3 * (phi - 1) * trQ2
        if part != 1:
            for i in range(5):
                g.x4[i, :] += (2 if i != 2 else 6) * self.wv1 * self.aux.c_lap * x.x4[i, :]

        # chain rule: apply (adjoint) DST to value derivatives
        g1 = g_v.sine_trans()
        g.x += g1.x
        if proj:
            self.project(g, 0)
        return g

    def grad_vec(self, x: np.ndarray, N, proj=True):
        """Evaluate gradient on vector input
        To prevent roundoff errors, always project input"""
        if proj:
            x = self.project_vec(x, N, self.conf.v0)
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

    def diffusion(self, X: LCState_s, proj=False):
        """The diffusion operator (in Q and phi) induced by the LdG energy
        which is composed of linear terms in the gradient"""
        N = X.N
        if self.aux is None:
            self.aux = LCFunc_s.LCAux(N)
        h3 = 1. / (N + 1)**3
        xv = X.sine_trans()  # function values
        q1, q2, q3, q4, q5, phi = \
            xv.q1, xv.q2, xv.q3, xv.q4, xv.q5, xv.phi

        def diff_op(dx: np.ndarray):
            nonlocal self, N, q1, q2, q3, q4, q5, phi, proj
            dX = LCState_s(N, dx, copy=True)
            dXvx = dX.xdiff(aux_k1=self.aux.k1)  # derivatives
            dXvy = dX.ydiff(aux_k2=self.aux.k2)
            dXvz = dX.zdiff(aux_k3=self.aux.k3)
            g = LCState_s(N)

            # Q-diffusion from LdG elastic and void
            # LdG elastic (now with mask, so evaluate with adjoints)
            for i in range(5):
                g.x4[i, :] = (2 if i != 2 else 6) * self.we * h3 * np.pi * (
                        self.aux.k1 * padded_dct_then_dst(phi**2 * dXvx.x4[i, :], axis=0)
                        + self.aux.k2 * padded_dct_then_dst(phi**2 * dXvy.x4[i, :], axis=1)
                        + self.aux.k3 * padded_dct_then_dst(phi**2 * dXvz.x4[i, :], axis=2)
                )
            # mixing Laplacian
            g.phi[:] = self.wp1 * self.aux.c_lap * dX.phi
            # Diffusion of phi in anchoring energy
            x1, x2, x3 = Qtimes(q1, q2, q3, q4, q5, dXvx.phi, dXvy.phi, dXvz.phi)
            d_a0_d_phix, d_a0_d_phiy, d_a0_d_phiz = \
                Qtimes(q1, q2, q3, q4, q5,
                       2 * x1 + self.sp * 4 / 3 * dXvx.phi,
                       2 * x2 + self.sp * 4 / 3 * dXvy.phi,
                       2 * x3 + self.sp * 4 / 3 * dXvz.phi)
            g.phi[:] += self.wa * h3 * np.pi * \
                        (self.aux.k1 * padded_dct_then_dst(d_a0_d_phix, axis=0)
                         + self.aux.k2 * padded_dct_then_dst(d_a0_d_phiy, axis=1)
                         + self.aux.k3 * padded_dct_then_dst(d_a0_d_phiz, axis=2))
            g.phi[:] += self.wa * self.sp**2 * 2 / 9 * self.aux.c_lap * dX.phi
            # void
            for i in range(5):
                g.x4[i, :] += (2 if i != 2 else 6) * self.wv1 * self.aux.c_lap * X.x4[i, :]

            if proj:
                # g must also satisfy constraint
                self.project(g, 0)
            return g.x

        return LinearOperator(dtype=float, shape=(6 * N**3, 6 * N**3),
                              matvec=diff_op)

    def hess(self, x: LCState_s):
        """Compute Hessian as implicit matrix (LinearOperator instance)
        Computes first variation of gradient from variation in x"""

        def _matvec(v):
            nonlocal self, x, N
            x_v = x.sine_trans()  # values
            dx = view_as_lc(v, N)  # variation
            dx_v = dx.sine_trans()  # variation values
            h3 = 1. / (N + 1)**3
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
                          ((2 * x_v.phi - 1) * (x_v.phi - .5)
                           + x_v.phi * (x_v.phi - 1)) * dx_v.phi
            # anchoring
            phix = padded_dct_then_dst(self.aux.k1 * np.pi * x.phi, axis=0)
            phiy = padded_dct_then_dst(self.aux.k2 * np.pi * x.phi, axis=1)
            phiz = padded_dct_then_dst(self.aux.k3 * np.pi * x.phi, axis=2)
            dphix = padded_dct_then_dst(self.aux.k1 * np.pi * dx.phi, axis=0)
            dphiy = padded_dct_then_dst(self.aux.k2 * np.pi * dx.phi, axis=1)
            dphiz = padded_dct_then_dst(self.aux.k3 * np.pi * dx.phi, axis=2)
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
                         (h3 * np.pi * (self.aux.k1 * padded_dct_then_dst(dd_a0_d_phix, axis=0)
                                        + self.aux.k2 * padded_dct_then_dst(dd_a0_d_phiy, axis=1)
                                        + self.aux.k3 * padded_dct_then_dst(dd_a0_d_phiz, axis=2))
                          + self.sp**2 * 2 / 9 * self.aux.c_lap * dx.phi)
            # void
            _1mphi2 = (1 - x_v.phi)**2
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
        return LinearOperator(dtype=float, shape=(6 * N**3, 6 * N**3),
                              matvec=_matvec)


def test_grad(FF: LCFunc_s, X: LCState_s, n=5):
    for _ in range(n):
        v = np.random.randn(len(X.x))
        v /= norm(v)
        for eps in 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7:
            dfdv = (FF.energy_vec(X.x + eps * v, X.N, proj=False)
                    - FF.energy_vec(X.x - eps * v, X.N, proj=False)) / 2 / eps
            gdotv = np.dot(FF.grad_vec(X.x, X.N, proj=False), v)
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


def generate_x(N):
    X = LCState_s(N)
    # Random state X, but with regularization
    reg = np.exp(-(FF.aux.k1**2 + FF.aux.k2**2 + FF.aux.k3**2) / 2)
    X.q1[:] = np.random.randn(N, N, N) * reg
    X.q2[:] = np.random.randn(N, N, N) * reg
    X.q3[:] = np.random.randn(N, N, N) * reg
    X.q4[:] = np.random.randn(N, N, N) * reg
    X.q5[:] = np.random.randn(N, N, N) * reg
    X.phi[:] = np.random.randn(N, N, N) * reg
    return X


if __name__ == "__main__":
    N = 7
    FF = LCFunc_s()
    FF.get_aux(N)
    FF.reset_params(show=True)
    np.random.seed(142)
    X = generate_x(N)

    test_grad(FF, X)
    # test_hess(FF, X)
    for i in range(5):
        X = generate_x(N)
        G = FF.grad(X)
        G1 = FF.grad(X, part=1)
        G2 = view_as_lc(FF.diffusion(X) @ X.x, N)
        print(norm(G.x - G1.x - G2.x))
