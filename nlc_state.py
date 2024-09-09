import json
import numpy as np
from numpy.linalg import norm
import scipy.fft as fft

"""Definition of 3D NLC state LdG tensor by Fourier spectral method"""

class LCConfig(dict):
    """LdG+PhF model configuration object. Same as python dictionary"""

    def __init__(self, **kwargs):
        p_default = {'A': -3900, 'B': 6400, 'C': 3500, 'L': 4e-11,
                     'N': 23, 'lam': 1e-6, 'vol': 0.04,
                     'alpha': .01,
                     'wv': 15,
                     'wa': 5,
                     'wp': 1}
        p_default.update(kwargs)
        super().__init__(**p_default)


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


class LCState_s:
    """State variable of LdG liquid crystal Q and phase field φ.
    Both expressed with sines."""

    def __init__(self, N: int, x=None, copy=False):
        self.N = N  # freq. of sines = #collocation
        foo = N ** 3
        if x is None:
            self.x = np.zeros(6 * foo)
        else:
            assert x.shape[0] == 6 * foo
            if not copy:
                self.x = x[:]  # DO NOT copy values
            else:
                self.x = np.copy(x)
        self.q1 = self.x[0:foo].reshape([N, N, N])
        self.q2 = self.x[foo:foo * 2].reshape([N, N, N])
        self.q3 = self.x[foo * 2:foo * 3].reshape([N, N, N])
        self.q4 = self.x[foo * 3:foo * 4].reshape([N, N, N])
        self.q5 = self.x[foo * 4:foo * 5].reshape([N, N, N])
        self.phi = self.x[foo * 5:].reshape([N, N, N])

    def values(self, sz=None):
        """Compute function values with type-I DST (identical in form as IDST).
        May compute on a finer grid (give the sz argument)"""
        if sz is None:
            sz = self.N
        sz = (sz, sz, sz)
        # DST with manual scaling (cancel factor 2 in standard form)
        q1 = .125 * fft.dstn(self.q1, type=1, s=sz)
        q2 = .125 * fft.dstn(self.q2, type=1, s=sz)
        q3 = .125 * fft.dstn(self.q3, type=1, s=sz)
        q4 = .125 * fft.dstn(self.q4, type=1, s=sz)
        q5 = .125 * fft.dstn(self.q5, type=1, s=sz)
        phi = .125 * fft.dstn(self.phi, type=1, s=sz)
        return q1, q2, q3, q4, q5, phi

    def sine_trans(self, sz=None):
        """Compute function values with type-I DST (identical in form as IDST).
        Return another LCState object containing function values."""
        if sz is None:
            sz = self.N
        x1 = LCState_s(sz)
        x1.q1[:] = .125 * fft.dstn(self.q1, type=1, s=(sz, sz, sz))
        x1.q2[:] = .125 * fft.dstn(self.q2, type=1, s=(sz, sz, sz))
        x1.q3[:] = .125 * fft.dstn(self.q3, type=1, s=(sz, sz, sz))
        x1.q4[:] = .125 * fft.dstn(self.q4, type=1, s=(sz, sz, sz))
        x1.q5[:] = .125 * fft.dstn(self.q5, type=1, s=(sz, sz, sz))
        x1.phi[:] = .125 * fft.dstn(self.phi, type=1, s=(sz, sz, sz))
        return x1

    def phi_values(self, sz=None):
        if sz is None:
            sz = self.N
        return .125 * fft.dstn(self.phi, type=1, s=(sz, sz, sz))

    def values_x(self, x, y, z, phi_only=False):
        """Return function values given (arbitrary) points"""
        if not phi_only:
            q1 = np.zeros(x.shape)
            q2 = np.zeros(x.shape)
            q3 = np.zeros(x.shape)
            q4 = np.zeros(x.shape)
            q5 = np.zeros(x.shape)
        phi = np.zeros(x.shape)
        base = np.zeros_like(x)
        for k1 in range(1, self.N + 1):
            for k2 in range(1, self.N + 1):
                for k3 in range(1, self.N + 1):
                    base[:] = np.sin(k1 * np.pi * x) * np.sin(k2 * np.pi * y) * np.sin(k3 * np.pi * z)
                    if not phi_only:
                        q1 += np.sin(self.q1[k1 - 1, k2 - 1, k3 - 1]) * base
                        q2 += np.sin(self.q2[k1 - 1, k2 - 1, k3 - 1]) * base
                        q3 += np.sin(self.q3[k1 - 1, k2 - 1, k3 - 1]) * base
                        q4 += np.sin(self.q4[k1 - 1, k2 - 1, k3 - 1]) * base
                        q5 += np.sin(self.q5[k1 - 1, k2 - 1, k3 - 1]) * base
                    phi += np.sin(self.phi[k1 - 1, k2 - 1, k3 - 1]) * base
        if not phi_only:
            return q1, q2, q3, q4, q5, phi
        return phi

    def proj_phi(self, v0):
        """Project phi to `v0`"""
        foo = 2. / np.pi / np.arange(1, self.N + 1)
        foo[1::2] = 0
        int_of_sines = foo.reshape([self.N, 1, 1]) * foo.reshape([1, self.N, 1]) * foo.reshape([1, 1, self.N])
        s = np.sum(foo ** 2)
        self.phi -= (np.sum(int_of_sines * self.phi) - v0) * int_of_sines / s ** 3


def view_as_lc(x0: np.ndarray, N):
    """Convert (or `view` in numpy terminology) arrays to LCState_s"""
    return LCState_s(N, x0.ravel())


def save_lc(filename, X: LCState_s):
    """Save LC state (preserve size N for later retrieval)"""
    np.save(filename, X.x.reshape([6, X.N, X.N, X.N]))


def load_lc(filename):
    """Load state from .npy file and convert to LCState_s class"""
    x = np.load(filename)
    return view_as_lc(x.ravel(), x.shape[1])