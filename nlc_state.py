import numpy as np
import scipy.fft as fft

"""Definition of 3D NLC state LdG tensor by Fourier spectral method"""


def cos_trans(x, axis):
    N1, N2, N3 = x.shape
    if axis == 0:
        return .125 * fft.dstn(fft.dctn(
            np.concatenate([np.zeros([1, N2, N3]), x], axis=0),
            type=1, s=N1 + 2, axes=0)[1:N1 + 1, :, :],
                               type=1, axes=(1, 2))
    elif axis == 1:
        return .125 * fft.dstn(fft.dctn(
            np.concatenate([np.zeros([N1, 1, N3]), x], axis=1),
            type=1, s=N2 + 2, axes=1)[:, 1:N2 + 1, :],
                               type=1, axes=(0, 2))
    elif axis == 2:
        return .125 * fft.dstn(fft.dctn(
            np.concatenate([np.zeros([N1, N2, 1]), x], axis=2),
            type=1, s=N3 + 2, axes=2)[:, :, 1:N3 + 1],
                               type=1, axes=(0, 1))


class LCState_s:
    """State variable of LdG liquid crystal Q and phase field φ.
    Both expressed with sines."""

    def __init__(self, N: int, x=None, copy=False):
        self.N = N  # freq. of sines = #collocation
        foo = N**3
        if x is None:
            self.x = np.zeros(6 * foo)
        else:
            assert x.shape[0] == 6 * foo
            if not copy:
                self.x = x[:]  # DO NOT copy values
            else:
                self.x = np.copy(x)
        self.x4 = self.x.reshape([6, N, N, N])  # 4-dimensional array structure
        self.q1 = self.x[0:foo].reshape([N, N, N])
        self.q2 = self.x[foo:foo * 2].reshape([N, N, N])
        self.q3 = self.x[foo * 2:foo * 3].reshape([N, N, N])
        self.q4 = self.x[foo * 3:foo * 4].reshape([N, N, N])
        self.q5 = self.x[foo * 4:foo * 5].reshape([N, N, N])
        self.q = self.x[0:foo * 5].reshape([5, N, N, N])  # reference to all of Q
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
        for i in range(6):
            x1.x4[i, :] = .125 * fft.dstn(self.x4[i, :], type=1, s=(sz, sz, sz))
        return x1

    def phi_values(self, sz=None):
        if sz is None:
            sz = self.N
        return .125 * fft.dstn(self.phi, type=1, s=(sz, sz, sz))

    def xdiff(self, aux_k1=None):
        """Compute value of x-derivatives at corresponding grid points
        through sine and cosine transforms"""
        N = self.N
        # Index of x frequency
        # Reshape into (N,1,1) to broadcast
        k1 = aux_k1 if aux_k1 is not None else \
            np.arange(1, N + 1).reshape([N, 1, 1])
        xv = LCState_s(self.N)
        for i in range(6):
            xv.x4[i, :] = cos_trans(np.pi * k1 * self.x4[i, :], axis=0)
        return xv

    def ydiff(self, aux_k2=None):
        """Compute value of y-derivatives"""
        N = self.N
        k2 = aux_k2 if aux_k2 is not None else \
            np.arange(1, N + 1).reshape([1, N, 1])
        xv = LCState_s(N)
        for i in range(6):
            xv.x4[i, :] = cos_trans(np.pi * k2 * self.x4[i, :], axis=1)
        return xv

    def zdiff(self, aux_k3=None):
        """Compute value of z-derivatives"""
        N = self.N
        k3 = aux_k3 if aux_k3 is not None else \
            np.arange(1, N + 1).reshape([1, 1, N])
        xv = LCState_s(N)
        for i in range(6):
            xv.x4[i, :] = cos_trans(np.pi * k3 * self.x4[i, :], axis=2)
        return xv

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
        s = np.sum(foo**2)
        self.phi -= (np.sum(int_of_sines * self.phi) - v0) * int_of_sines / s**3


def resize_lc(X: LCState_s, N1):
    """Resize to a different frequency by padding 0's or truncating."""
    if X.N < N1:
        # N is smaller: pad 0's in high-freq components
        w = N1 - X.N
        x1 = np.pad(X.x4, [(0, 0), (0, w), (0, w), (0, w)])
    else:
        # N is larger: truncate
        x1 = np.copy(X.x4[:, 0:N1, 0:N1, 0:N1])
    return LCState_s(N1, x1.ravel())


def view_as_lc(x0: np.ndarray, N):
    """Convert (or `view` in numpy terminology) arrays to LCState_s"""
    return LCState_s(N, x0.ravel())


def save_lc(filename, X: LCState_s):
    """Save LC state (preserve size N for later retrieval)"""
    np.save(filename, X.x4)


def load_lc(filename, resize=0):
    """Load state from .npy file and convert to LCState_s class"""
    x = np.load(filename)
    X0 = view_as_lc(x, x.shape[1])
    if resize:
        return resize_lc(X0, resize)
    else:
        return X0


if __name__ == "__main__":
    np.random.seed(20241003)
    N = 7
    X = LCState_s(N)
    Y = LCState_s(N)
    k1 = np.arange(1, N + 1).reshape([N, 1, 1])
    # Adjoint of the differentiation operator
    for i in range(10):
        X.x[:] = np.random.randn(6 * N**3)
        Y.x[:] = np.random.randn(6 * N**3)
        Xx = X.xdiff()
        print(np.sum(Xx.q1 * Y.q1),
              np.sum(X.q1 * np.pi * k1 * cos_trans(Y.q1, axis=0)))

    X1 = X.sine_trans(sz=31)
