"""Computations of the LC state"""
from nlc_state import *
import json
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator


# Some computation on the Q tensor
@profiler
def Qtimes(q1, q2, q3, q4, q5, v1, v2, v3):
    """Compute Q*v, the result of which collected into a 3*N*N*N ndarray
    ** The shapes of q1~q5 and v1~v3 must be identical respectively;
    the multiplication between q[1-5] and v[1-3] should be permitted by numpy. **"""
    return np.array([(q1 - q3) * v1 + q2 * v2 + q4 * v3,
                     q2 * v1 + (-q1 - q3) * v2 + q5 * v3,
                     q4 * v1 + q5 * v2 + 2 * q3 * v3])


@profiler
def dQtimesv(q1, q2, q3, q4, q5, v1, v2, v3,
             dq1, dq2, dq3, dq4, dq5, dv1, dv2, dv3):
    """Compute variation of Q*v (by Leibniz's rule)"""
    return (q1 - q3) * dv1 + q2 * dv2 + q4 * dv3 \
           + (dq1 - dq3) * v1 + dq2 * v2 + dq4 * v3, \
           q2 * dv1 + (-q1 - q3) * dv2 + q5 * dv3 \
           + dq2 * v1 + (-dq1 - dq3) * v2 + dq5 * v3, \
           q4 * dv1 + q5 * dv2 + 2 * q3 * dv3 \
           + dq4 * v1 + dq5 * v2 + 2 * dq3 * v3


@profiler
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


@profiler
def trace_Q2(q1, q2, q3, q4, q5):
    return 2 * (q1 * q1 + q2 * q2 + 3 * q3 * q3 + q4 * q4 + q5 * q5)


@profiler
def trace_Q3(q1, q2, q3, q4, q5):
    return 3 * (2 * q3 * (q3 * q3 - q1 * q1) + 2 * q2 * q4 * q5
                - (q1 - q3) * q5 * q5 + (q1 + q3) * q4 * q4 - 2 * q3 * q2 * q2)


@profiler
def biaxiality(q1, q2, q3, q4, q5):
    """tr(Q^3)^2/tr(Q^2)^3, which is between 0 and 1/6"""
    return 1 - 6 * trace_Q3(q1, q2, q3, q4, q5)**2 / (trace_Q2(q1, q2, q3, q4, q5)**3 + 1e-14)


@profiler
def Q_eigval(q1, q2, q3, q4, q5, largest=False):
    # coefficients
    p = -(q1**2 + q2**2 + 3 * q3**2 + q4**2 + q5**2)
    q = -trace_Q3(q1, q2, q3, q4, q5) / 3
    return solve_cubic(np.zeros_like(q1), p, q, largest)


class LCConfig(dict):
    """LdG+PhF model configuration object. Same as python dictionary"""
    KEYS = ['A', 'B', 'C', 'L', 'lam', 'v0', 'eps', 'omega', 'wp', 'wv']

    def __init__(self, **kwargs):
        p_default = {'A': -3900, 'B': 6400, 'C': 3500, 'L': 4e-11,
                     'lam': 1e-6, 'v0': 0.04,
                     'eps': .01, 'omega': 5, 'wp': 1, 'wv': 1}
        p_default.update(kwargs)
        super().__init__(**p_default)
        for key, value in p_default.items():
            setattr(self, key, value)

    def __setattr__(self, __name: str, __value):
        if __name not in LCConfig.KEYS:
            raise KeyError("Invalid key %s for LCConfig; key must be in %a" % (__name,
                                                                               LCConfig.KEYS))
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
        self.conf = LCConfig(A=-1500, lam=2e-7, v0=0.1,
                             eps=0.005, omega=20, wp=1, wv=0.5)
        self.conf.update(kwargs)
        self.reset_params(**self.conf)
        self.aux = None

    @profiler
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
            print("   ⎪   φ²[ λ·(A·|Q|²/2C - B·tr(Q³)/3C + |Q|⁴/4 - bmin) + λ⁻¹·|∇Q|²/2 ] dx")
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

    @profiler
    def reset_conf(self, conf: LCConfig, show=False):
        self.conf.update(conf)
        self.reset_params(show, **conf)

    @profiler
    def export_conf(self, fn):
        save_lc_config(fn, self.conf)

    class LCAux:
        """Auxiliary variables that are used repeatedly"""

        @profiler
        def __init__(self, N: int):
            self.N = N
            one2n = np.arange(1, N + 1)
            k1, k2, k3 = np.meshgrid(one2n, one2n, one2n, indexing='ij')
            # grid of indices
            # save compactly, broadcast when used
            self.k1 = one2n.reshape([N, 1, 1])
            self.k2 = one2n.reshape([1, N, 1])
            self.k3 = one2n.reshape([1, 1, N])
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
            self.ios3_h1 = np.sum(self.ios3**2 / self.c_lap)

    @profiler
    def get_aux(self, N: int):
        self.aux = LCFunc_s.LCAux(N)

    @profiler
    def volume(self, x: LCState_s):
        if self.aux is None or self.aux.N != x.N:
            self.aux = LCFunc_s.LCAux(x.N)
        return np.sum(self.aux.ios3 * x.phi)

    @profiler
    def vol_con_normal(self, phi, v0, metric="h1"):
        """Return normal component to the volume constraint"""
        if self.aux is None or self.aux.N != phi.shape[0]:
            self.aux = LCFunc_s.LCAux(phi.shape[0])
        ios3 = self.aux.ios3
        r = self.aux.ios3_norm2 if metric == "l2" else self.aux.ios3_h1
        if metric == "l2":
            return (np.sum(ios3 * phi) - v0) * ios3 / r
        else:
            return (np.sum(ios3 * phi) - v0) / r * (ios3 / self.aux.c_lap)

    @profiler
    def project(self, phi: np.ndarray, v0, metric="h1"):
        """Project phi to v0 by modifing in-place"""
        phi -= self.vol_con_normal(phi, v0, metric=metric)

    @profiler
    def project_vec(self, x: np.ndarray, N, v0, metric="h1"):
        """Project phi to v0, but with vector input.
        Returns new vector"""
        if self.aux is None or self.aux.N != N:
            self.aux = LCFunc_s.LCAux(N)
        xp = np.copy(x).reshape([6, N, N, N])
        xp[5] -= self.vol_con_normal(xp[5], v0, metric)
        return xp.ravel()

    @profiler
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

    @profiler
    def energy_vec(self, x: np.ndarray, N, proj="h1"):
        """To prevent roundoff error, project x to v0 when input is vector"""
        if proj:
            x = self.project_vec(x, N, self.conf.v0, metric=proj)
        return self.energy(view_as_lc(x, N))

    @profiler
    def grad(self, x: LCState_s, part=2, proj=False):
        """Gradient of functional

        ** arguments **

        `part`: if part==0, evaluate only the linear terms (Laplacians)
                if part==1, evaluate only the nonlinear terms
                if part==2, evaluate the whole gradient
        `proj`: whether to project gradient; equals the metric of the underlying space"""
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

        trQ3 = trace_Q3(q1, q2, q3, q4, q5)
        bulk = self.wb2 / 2 * trQ2 + self.wb3 / 3 * trQ3 + .25 * trQ2**2 - self.bmin
        elastic = (xvx.q1**2 + xvx.q2**2 + 3 * xvx.q3**2 + xvx.q4**2 + xvx.q5**2
                   + xvy.q1**2 + xvy.q2**2 + 3 * xvy.q3**2 + xvy.q4**2 + xvy.q5**2
                   + xvz.q1**2 + xvz.q2**2 + 3 * xvz.q3**2 + xvz.q4**2 + xvz.q5**2)
        x1, x2, x3 = Qtimes(q1, q2, q3, q4, q5, xvx.phi, xvy.phi, xvz.phi)  # Q∇φ
        if part != 1:
            pass
            # LdG elastic (now with mask, so evaluate with adjoints)
            for i in range(5):
                g.x4[i, :] = (2 if i != 2 else 6) * self.we * h3 * np.pi * (
                        self.aux.k1 * cos_trans(phi**2 * xvx.x4[i, :], axis=0)
                        + self.aux.k2 * cos_trans(phi**2 * xvy.x4[i, :], axis=1)
                        + self.aux.k3 * cos_trans(phi**2 * xvz.x4[i, :], axis=2)
                )
                # Laplacian from void
                g.x4[i, :] += (2 if i != 2 else 6) * self.wv1 * self.aux.c_lap * x.x4[i, :]
            # mixing Laplacian and a term from anchoring energy
            g.phi[:] = (self.wp1 + self.wa * self.sp**2 * 2 / 9) * self.aux.c_lap * x.phi
            # diffusion of phi in anchoring energy
            d_a0_d_phix, d_a0_d_phiy, d_a0_d_phiz = \
                Qtimes(q1, q2, q3, q4, q5,
                       2 * x1 + self.sp * 4 / 3 * xvx.phi,
                       2 * x2 + self.sp * 4 / 3 * xvy.phi,
                       2 * x3 + self.sp * 4 / 3 * xvz.phi)
            g.phi[:] += self.wa * h3 * np.pi * \
                        (self.aux.k1 * cos_trans(d_a0_d_phix, axis=0)
                         + self.aux.k2 * cos_trans(d_a0_d_phiy, axis=1)
                         + self.aux.k3 * cos_trans(d_a0_d_phiz, axis=2))
        if part != 0:
            # LdG bulk over Q
            QpA_C = trQ2 + self.wb2
            g_v.q1[:] += self.wb * h3 * phi**2 * \
                         (2 * QpA_C * q1 + self.wb3 * (-4 * q3 * q1 - q5 * q5 + q4 * q4))
            g_v.q3[:] += self.wb * h3 * phi**2 * \
                         (6 * QpA_C * q3 + self.wb3 * (6 * q3 * q3 - 2 * q1 * q1
                                                       + q5 * q5 + q4 * q4 - 2 * q2 * q2))
            g_v.q2[:] += self.wb * h3 * phi**2 * \
                         (2 * QpA_C * q2 + self.wb3 * (2 * q4 * q5 - 4 * q3 * q2))
            g_v.q4[:] += self.wb * h3 * phi**2 * \
                         (2 * QpA_C * q4 + self.wb3 * (2 * q2 * q5 + 2 * (q1 + q3) * q4))
            g_v.q5[:] += self.wb * h3 * phi**2 * \
                         (2 * QpA_C * q5 + self.wb3 * (2 * q2 * q4 - 2 * (q1 - q3) * q5))
            # elastic and bulk over phi
            g_v.phi[:] = 2 * h3 * phi * (self.we * elastic + self.wb * bulk)
            # mixing energy
            g_v.phi[:] += 4 * self.wp0 * h3 * \
                          phi * (phi - 1) * (phi - .5)
            # anchoring over Q
            g_v.q1[:] += h3 * self.wa * \
                         (2 * (x1 * xvx.phi - x2 * xvy.phi)
                          + self.sp * 2 / 3 * (xvx.phi**2 - xvy.phi**2))
            g_v.q3[:] += h3 * self.wa * \
                         (2 * (-x1 * xvx.phi - x2 * xvy.phi + 2 * x3 * xvz.phi)
                          + self.sp * 2 / 3 * (-xvx.phi**2 - xvy.phi**2 + 2 * xvz.phi**2))
            g_v.q2[:] += h3 * self.wa * \
                         (2 * (x1 * xvy.phi + x2 * xvx.phi)
                          + self.sp * 4 / 3 * xvx.phi * xvy.phi)
            g_v.q4[:] += h3 * self.wa * \
                         (2 * (x1 * xvz.phi + x3 * xvx.phi)
                          + self.sp * 4 / 3 * xvx.phi * xvz.phi)
            g_v.q5[:] += h3 * self.wa * \
                         (2 * (x2 * xvz.phi + x3 * xvy.phi)
                          + self.sp * 4 / 3 * xvy.phi * xvz.phi)
            # void
            g_v.phi[:] += self.wv * h3 * (phi - 1) * trQ2
            for i in range(5):
                g_v.x4[i, :] += (2 if i != 2 else 6) * self.wv * h3 * (1 - phi)**2 * xv.x4[i, :]

            # chain rule: apply (adjoint) DST to value derivatives
            g.x += g_v.sine_trans().x

        if proj == "h1":
            g.phi /= self.aux.c_lap  # convert to gradient under H1 metric
            g.phi -= self.vol_con_normal(g.phi, 0, metric="h1")
        elif proj == "l2":
            g.phi -= self.vol_con_normal(g.phi, 0, metric="l2")
        return g

    @profiler
    def grad_vec(self, x: np.ndarray, N, proj="h1"):
        """Evaluate gradient with vector input"""
        return self.grad(view_as_lc(x, N), proj=proj).x

    @profiler
    def gradQ(self, x: LCState_s):
        """Only the Q-gradient"""
        g = self.grad(x)
        g.phi[:] = 0.  # clear the phi gradient
        return g

    @profiler
    def gradQ_vec(self, x: np.ndarray, N):
        return self.gradQ(view_as_lc(x, N)).x

    @profiler
    def hess(self, x: LCState_s, part="all", proj=False):
        """Compute Hessian as implicit matrix (LinearOperator instance)

        E = E1(∇Q,φ) + E2(Q,∇φ) + E3(Q,φ)
        
        2nd variation of E1 over Q, and E2 over φ are linear diffusion operators
        rest of E1 and E2 are nonlinear terms
        variation of E3 is entirely local (i.e. concerning values only)
        """
        N = x.N
        if self.aux is None:
            self.aux = LCFunc_s.LCAux(N)
        h3 = 1. / (N + 1)**3
        xv = x.sine_trans()  # function values
        q1, q2, q3, q4, q5, phi = \
            xv.q1, xv.q2, xv.q3, xv.q4, xv.q5, xv.phi
        trQ2 = trace_Q2(q1, q2, q3, q4, q5)
        trQ3 = trace_Q3(q1, q2, q3, q4, q5)
        bulk = self.wb2 / 2 * trQ2 + self.wb3 / 3 * trQ3 + .25 * trQ2**2 - self.bmin
        QpA_C = trQ2 + self.wb2
        gvb1 = 2 * QpA_C * q1 + self.wb3 * (-4 * q3 * q1 - q5 * q5 + q4 * q4)
        gvb2 = 2 * QpA_C * q2 + self.wb3 * (2 * q4 * q5 - 4 * q3 * q2)
        gvb3 = 6 * QpA_C * q3 + self.wb3 * (6 * q3 * q3 - 2 * q1 * q1
                                            + q5 * q5 + q4 * q4 - 2 * q2 * q2)
        gvb4 = 2 * QpA_C * q4 + self.wb3 * (2 * q2 * q5 + 2 * (q1 + q3) * q4)
        gvb5 = 2 * QpA_C * q5 + self.wb3 * (2 * q2 * q4 - 2 * (q1 - q3) * q5)

        xvx = x.xdiff(aux_k1=self.aux.k1)
        xvy = x.ydiff(aux_k2=self.aux.k2)
        xvz = x.zdiff(aux_k3=self.aux.k3)
        phix = xvx.phi
        phiy = xvy.phi
        phiz = xvz.phi
        x1, x2, x3 = Qtimes(q1, q2, q3, q4, q5, phix, phiy, phiz)
        elastic = xvx.q1**2 + xvx.q2**2 + 3 * xvx.q3**2 + xvx.q4**2 + xvx.q5**2 \
                  + xvy.q1**2 + xvy.q2**2 + 3 * xvy.q3**2 + xvy.q4**2 + xvy.q5**2 \
                  + xvz.q1**2 + xvz.q2**2 + 3 * xvz.q3**2 + xvz.q4**2 + xvz.q5**2

        @profiler
        def dd_E1(dq, dphiv=None, linear=True):
            """2nd-order variation of E1 (the ∇Q terms)
            linear==True : diffusion part in Q only,
            linear==False : full variation
            """
            nonlocal self, phi, elastic, xvx, xvy, xvz, N, h3
            dq.shape = (5, N, N, N)
            d_elastic = np.zeros([N, N, N])
            y = np.zeros([5, N, N, N])
            for i in range(5):
                # derivatives of dQ
                dqx_v = np.pi * cos_trans(self.aux.k1 * dq[i], axis=0)
                dqy_v = np.pi * cos_trans(self.aux.k2 * dq[i], axis=1)
                dqz_v = np.pi * cos_trans(self.aux.k3 * dq[i], axis=2)
                # elastic energy and laplacian from void
                if linear:
                    # just terms of dQ
                    y[i] = (2 if i != 2 else 6) * (self.we * h3 * np.pi *
                                                   (self.aux.k1 * cos_trans(phi**2 * dqx_v, axis=0)
                                                    + self.aux.k2 * cos_trans(phi**2 * dqy_v, axis=1)
                                                    + self.aux.k3 * cos_trans(phi**2 * dqz_v, axis=2))
                                                   + self.wv1 * self.aux.c_lap * dq[i])
                else:
                    # variation in elastic energy |∇Q|^2
                    d_elastic += (2 if i != 2 else 6) * (xvx.x4[i] * dqx_v
                                                         + xvy.x4[i] * dqy_v
                                                         + xvz.x4[i] * dqz_v)
                    dphiv.shape = (N, N, N)
                    # include terms of dphi as well
                    y[i] = (2 if i != 2 else 6) * (
                            self.we * h3 * np.pi * (
                            self.aux.k1 * cos_trans(phi * (phi * dqx_v + 2 * xvx.x4[i] * dphiv), axis=0)
                            + self.aux.k2 * cos_trans(phi * (phi * dqy_v + 2 * xvy.x4[i] * dphiv), axis=1)
                            + self.aux.k3 * cos_trans(phi * (phi * dqz_v + 2 * xvz.x4[i] * dphiv), axis=2))
                            + self.wv1 * self.aux.c_lap * dq[i])
            if not linear:
                # 2nd variation of the phi-variation in function values
                z = 2 * self.we * h3 * (dphiv * elastic + phi * d_elastic)
                return y, z
            return y

        @profiler
        def dd_E2(dphi, dQv=None, linear=True):
            """2nd-order variation of E2 (the ∇φ terms)
            linear==True : diffusion part in φ only
            linear==False : full variation
            """
            nonlocal self, xv, N, h3
            nonlocal q1, q2, q3, q4, q5, phix, phiy, phiz
            nonlocal x1, x2, x3
            dphi.shape = (N, N, N)
            # mixing laplacian
            y = (self.wp1 + self.wa * self.sp**2 * 2 / 9) * self.aux.c_lap * dphi
            # anchoring
            dphix_v = np.pi * cos_trans(self.aux.k1 * dphi, axis=0)
            dphiy_v = np.pi * cos_trans(self.aux.k2 * dphi, axis=1)
            dphiz_v = np.pi * cos_trans(self.aux.k3 * dphi, axis=2)
            dx1_lin, dx2_lin, dx3_lin = \
                Qtimes(q1, q2, q3, q4, q5, dphix_v, dphiy_v, dphiz_v)
            # linear part of d(anchor)/d(∇φ)
            dd_a0_d_phix, dd_a0_d_phiy, dd_a0_d_phiz = \
                Qtimes(q1, q2, q3, q4, q5,
                       2 * dx1_lin + self.sp * 4 / 3 * dphix_v,
                       2 * dx2_lin + self.sp * 4 / 3 * dphiy_v,
                       2 * dx3_lin + self.sp * 4 / 3 * dphiz_v
                       )
            if not linear:
                dQv.shape = (5, N, N, N)
                dx1_nonlin, dx2_nonlin, dx3_nonlin = \
                    Qtimes(dQv[0], dQv[1], dQv[2], dQv[3], dQv[4], phix, phiy, phiz)
                # the rest of d(anchor)/d(∇φ)
                dd_a0_d_phix_nonlin, dd_a0_d_phiy_nonlin, dd_a0_d_phiz_nonlin = \
                    Qtimes(q1, q2, q3, q4, q5,
                           2 * dx1_nonlin, 2 * dx2_nonlin, 2 * dx3_nonlin) + \
                    Qtimes(dQv[0], dQv[1], dQv[2], dQv[3], dQv[4],
                           2 * x1 + self.sp * 4 / 3 * phix,
                           2 * x2 + self.sp * 4 / 3 * phiy,
                           2 * x3 + self.sp * 4 / 3 * phiz)
                dd_a0_d_phix += dd_a0_d_phix_nonlin
                dd_a0_d_phiy += dd_a0_d_phiy_nonlin
                dd_a0_d_phiz += dd_a0_d_phiz_nonlin
            y += self.wa * h3 * np.pi * \
                 (self.aux.k1 * cos_trans(dd_a0_d_phix, axis=0)
                  + self.aux.k2 * cos_trans(dd_a0_d_phiy, axis=1)
                  + self.aux.k3 * cos_trans(dd_a0_d_phiz, axis=2))
            if not linear:
                # 2nd variation of the Q-variation in function values
                dx1 = dx1_lin + dx1_nonlin
                dx2 = dx2_lin + dx2_nonlin
                dx3 = dx3_lin + dx3_nonlin
                z = h3 * 2 * self.wa * np.stack([
                    (dx1 * phix + x1 * dphix_v - dx2 * phiy - x2 * dphiy_v)
                    + self.sp * 2 / 3 * (phix * dphix_v - phiy * dphiy_v),
                    (dx1 * phiy + x1 * dphiy_v + dx2 * phix + x2 * dphix_v)
                    + self.sp * 2 / 3 * (dphix_v * phiy + phix * dphiy_v),
                    (-dx1 * phix - x1 * dphix_v - dx2 * phiy - x2 * dphiy_v
                     + 2 * (dx3 * phiz + x3 * dphiz_v))
                    + self.sp * 2 / 3 * (
                            -phix * dphix_v - phiy * dphiy_v + 2 * phiz * dphiz_v),
                    (dx1 * phiz + x1 * dphiz_v + dx3 * phix + x3 * dphix_v)
                    + self.sp * 2 / 3 * (dphix_v * phiz + phix * dphiz_v),
                    (dx2 * phiz + x2 * dphiz_v + dx3 * phiy + x3 * dphiy_v)
                    + self.sp * 2 / 3 * (dphiy_v * phiz + phiy * dphiz_v)
                ], axis=0)
                return y, z
            return y

        @profiler
        def _matvec(v):
            nonlocal proj, part
            nonlocal self, xv, N, h3
            nonlocal q1, q2, q3, q4, q5, phi, trQ2, QpA_C, bulk
            nonlocal gvb1, gvb2, gvb3, gvb4, gvb5
            dx = view_as_lc(np.copy(v), N)  # variation
            if proj:
                # Project input vector
                v_normal = self.vol_con_normal(dx.phi, 0, metric=proj)
                dx.phi -= v_normal
            dxv = dx.sine_trans()  # variation values
            dg = LCState_s(N)  # gradient variation

            if part == "diffusion":
                # Diffusion in Q from E1
                dg.x4[0:5] = dd_E1(dx.q, linear=True)
                # Diffusion in φ from E2
                dg.phi[:] = dd_E2(dx.phi, linear=True)
            elif part == "all":
                dg_v = LCState_s(N)  # gradient w.r.t. values variation
                dtrQ2 = 4 * (q1 * dxv.q1 + q2 * dxv.q2 + 3 * q3 * dxv.q3 + q4 * dxv.q4 + q5 * dxv.q5)
                # The energy E1(∇Q,φ)
                y, z = dd_E1(dx.q, dxv.phi, linear=False)
                dg.x4[0:5] += y
                dg_v.phi[:] += z
                # The energy E2(Q,∇φ)
                y, z = dd_E2(dx.phi, dxv.x4[0:5], linear=False)
                dg.phi[:] += y
                dg_v.x4[0:5] += z
                ## The energy E3(Q,φ) (local terms)
                # LdG bulk
                dg_v.q1[:] += 2 * self.wb * h3 * (phi**2 *
                                                  (QpA_C * dxv.q1 + dtrQ2 * q1
                                                   + self.wb3 * (-2 * (q3 * dxv.q1 + dxv.q3 * q1)
                                                                 - q5 * dxv.q5 + q4 * dxv.q4))
                                                  + gvb1 * phi * dxv.phi)
                dg_v.q3[:] += 2 * self.wb * h3 * (phi**2 *
                                                  (3 * (QpA_C * dxv.q3 + dtrQ2 * q3)
                                                   + self.wb3 * (6 * q3 * dxv.q3 - 2 * q1 * dxv.q1
                                                                 + q5 * dxv.q5 + q4 * dxv.q4
                                                                 - 2 * q2 * dxv.q2))
                                                  + gvb3 * phi * dxv.phi)
                dg_v.q2[:] += 2 * self.wb * h3 * (phi**2 *
                                                  (QpA_C * dxv.q2 + dtrQ2 * q2
                                                   + self.wb3 * ((q4 * dxv.q5 + q5 * dxv.q4)
                                                                 - 2 * (q3 * dxv.q2 + q2 * dxv.q3)))
                                                  + gvb2 * phi * dxv.phi)
                dg_v.q4[:] += 2 * self.wb * h3 * (phi**2 *
                                                  (QpA_C * dxv.q4 + dtrQ2 * q4
                                                   + self.wb3 * (q2 * dxv.q5 + q5 * dxv.q2
                                                                 + ((q1 + q3) * dxv.q4
                                                                    + q4 * (dxv.q1 + dxv.q3))))
                                                  + gvb4 * phi * dxv.phi)
                dg_v.q5[:] += 2 * self.wb * h3 * (phi**2 *
                                                  (QpA_C * dxv.q5 + dtrQ2 * q5
                                                   + self.wb3 * (q2 * dxv.q4 + q4 * dxv.q2
                                                                 - ((q1 - q3) * dxv.q5
                                                                    + q5 * (dxv.q1 - dxv.q3))))
                                                  + gvb5 * phi * dxv.phi)
                dbulk = gvb1 * dxv.q1 + gvb2 * dxv.q2 + gvb3 * dxv.q3 + gvb4 * dxv.q4 + gvb5 * dxv.q5
                dg_v.phi[:] += 2 * self.wb * h3 * (dbulk * phi + bulk * dxv.phi)
                # mixing double-well
                dg_v.phi[:] += 4 * self.wp0 * h3 * (3 * phi**2 - 3 * phi + 0.5) * dxv.phi
                # penalty from void
                for i in range(5):
                    dg_v.x4[i] += (2 if i != 2 else 6) * self.wv * h3 * (
                            (1 - phi)**2 * dxv.x4[i] + 2 * (phi - 1) * xv.x4[i] * dxv.phi)
                dg_v.phi[:] += self.wv * h3 * (dxv.phi * trQ2 + (phi - 1) * dtrQ2)

                # chain rule: apply (adjoint) DST to value derivatives
                dg.x += dg_v.sine_trans().x

            # Project gradient to meet volume constraint
            if proj == "l2":
                # Project output and keep normal component
                # Result equals P*H*P + v*v' (P is projection, v is normal vector)
                # self.project(dg.phi, 0, metric="l2")
                dg.phi += -self.vol_con_normal(dg.phi, 0, metric="l2") + v_normal
            elif proj == "h1":
                dg.phi /= self.aux.c_lap  # Convert to H1 gradient
                dg.phi += -self.vol_con_normal(dg.phi, 0, metric="h1") + v_normal
            # dg.phi += v_normal
            return dg.x

        return LinearOperator(dtype=float, shape=(6 * N**3, 6 * N**3),
                              matvec=_matvec)

    @profiler
    def hess_inv_approx(self, x: LCState_s):
        """Approximate Hessian inverse under the L2 metric"""
        if self.aux is None:
            self.get_aux(x.N)
        N = x.N
        xv = x.sine_trans()
        sp3 = self.sp / 3
        q1, q2, q3, q4, q5, phi = xv.x4
        # Average of φ^2
        phi2_avg = np.average(phi**2)
        # Diagonal elements of (Q+s/3)^2
        A11 = np.average((q1 - q3 + sp3)**2 + q2**2 + q4**2)
        # A12=np.average(2*(-q3+sp3)*q2+q4*q5)
        # A13=np.average((q1+q3+2*sp3)*q4+q2*q5)
        A22 = np.average(q2**2 + (-q1 - q3 + sp3)**2 + q5**2)
        # A23=np.average(q2*q4+(-q1+q3+2*sp3)*q5)
        A33 = np.average(q4**2 + q5**2 + (-2 * q3 + sp3)**2)
        # Approximate A^{ij} φ_i φ_j with diagonal operator
        #       wa*(A11*|φ_x|^2+A22*|φ_y|^2+A33*|φ_z|^2)
        # Its gradient equals
        # 2*wa* pi^2/8 * (A11*k1**2+A22*k2**2+A33*k3**2) * phi
        c_ell = self.wp1 * self.aux.c_lap + \
                .25 * self.wa * np.pi**2 * (A11 * self.aux.k1**2
                                            + A22 * self.aux.k2**2
                                            + A33 * self.aux.k3**2)

        def _matvec(v):
            nonlocal self
            nonlocal N, phi2_avg, A11, A22, A33, c_ell
            y = np.copy(v)
            y.shape = (6, N, N, N)
            # Approximate diffusion in Q with a Laplacian
            for i in range(5):
                y[i, :] /= (self.wv1 + self.we * phi2_avg) * self.aux.c_lap
            # Approximate diffusion in phi with a diagonal elliptic operator
            y[5, :] /= c_ell
            return y.ravel()

        return LinearOperator(dtype=float, shape=(6 * N**3, 6 * N**3),
                              matvec=_matvec)


def test_grad(FF: LCFunc_s, X: LCState_s, n=5):
    for _ in range(n):
        v = np.random.randn(len(X.x))
        v /= norm(v)
        for eps in 1e-2, 1e-3, 1e-4, 1e-5, 1e-6:
            dfdv = (FF.energy_vec(X.x + eps * v, X.N, proj=False)
                    - FF.energy_vec(X.x - eps * v, X.N, proj=False)) / 2 / eps
            gdotv = np.dot(FF.grad_vec(X.x, X.N, proj=False), v)
            print(np.log10(norm(dfdv - gdotv)), end=' ')
        print()


def test_hess(FF: LCFunc_s, X: LCState_s, n=5):
    H = FF.hess(X, proj="l2")
    for _ in range(n):
        v = np.random.randn(len(X.x))
        v = FF.project_vec(v, X.N, 0, metric="l2")
        v /= norm(v)
        for eps in 1e-2, 1e-3, 1e-4, 1e-5, 1e-6:
            dgdv = (FF.grad_vec(X.x + eps * v, X.N, proj="l2")
                    - FF.grad_vec(X.x - eps * v, X.N, proj="l2")) / 2 / eps
            Hv = H @ v
            print(np.log10(norm(dgdv - Hv)), end=' ')
        print()


def generate_x(N):
    X = LCState_s(N)
    # Random state X, but with regularization
    reg = 1. / (FF.aux.k1**2 + FF.aux.k2**2 + FF.aux.k3**2)
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
    # X = generate_x(N)
    X = view_as_lc(np.random.randn(6 * N**3), N)

    test_grad(FF, X)
    # test_hess(FF, X)
    # for i in range(5):
    #     X = generate_x(N)
    #     G = FF.grad(X)
    #     G1 = FF.grad(X, part=1)
    #     G2 = view_as_lc(FF.hess(X, part="diffusion") @ X.x, N)
    #     print(norm(G.x - G1.x - G2.x))
