import numpy as np
import scipy.fft as fft
import time
import os

"""Definition of 3D NLC state LdG tensor by Fourier spectral method"""

PROFILE = {}


def profiler(func):
    """A time metre of all subroutines"""

    def timed_func(*args, **kwargs):
        global PROFILE
        if func.__name__ not in PROFILE.keys():
            # Initialize profile
            PROFILE[func.__name__] = {"call": 0, "time": 0.}
        PROFILE[func.__name__]["call"] += 1
        t0 = time.time()
        res = func(*args, **kwargs)
        dt = time.time() - t0
        PROFILE[func.__name__]["time"] += dt
        return res

    return timed_func


def print_profile(width=None):
    global PROFILE
    if width is None:
        width = os.get_terminal_size().columns
    width = min(width, os.get_terminal_size().columns)
    maxt = max(PROFILE[x]['time'] for x in PROFILE)
    sorted_keys = sorted(PROFILE.keys(),
                         key=lambda x: PROFILE[x]['time'],
                         reverse=True)
    caps = [f"{x} {PROFILE[x]['call']:<7d} {PROFILE[x]['time']:>7.2f} " for x in sorted_keys]
    wcap = max(len(c) for c in caps)
    blocks = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]  # 1/8 steps
    for i in range(len(caps)):
        if width > wcap:
            r = PROFILE[sorted_keys[i]]['time'] / maxt * (width - wcap)
            print(f"{caps[i]:>{wcap}}" + "█" * int(r) + blocks[int(8 * r) % 8])
        else:
            print(f"{caps[i]:>{wcap}}")


@profiler
def sine_trans(x, s=None):
    """Get function value from coefficients by applying 3D sine transform.
    Manually scale by 1/8 to cancel factor 2 in standard form."""
    return .125 * fft.dstn(x, type=1, axes=(-3, -2, -1), s=s)  # Apply to last 3 axes


@profiler
def cos_trans(x, axis):
    """Apply cosine transform in one axis and sine transform in the others
    used when computing derivatives from Fourier coefficients."""
    N1, N2, N3 = x.shape
    if axis == 0 or axis == 'x':
        return .125 * fft.dstn(fft.dctn(
            np.concatenate([np.zeros([1, N2, N3]), x], axis=0),
            type=1, s=N1 + 2, axes=0)[1:N1 + 1, :, :],
                               type=1, axes=(1, 2))
    elif axis == 1 or axis == 'y':
        return .125 * fft.dstn(fft.dctn(
            np.concatenate([np.zeros([N1, 1, N3]), x], axis=1),
            type=1, s=N2 + 2, axes=1)[:, 1:N2 + 1, :],
                               type=1, axes=(0, 2))
    elif axis == 2 or axis == 'z':
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

    @profiler
    def values(self, sz=None):
        """Compute function values with type-I DST (identical in form as IDST).
        May compute on a finer grid (give the sz argument)"""
        if sz is None:
            sz = self.N
        sz = (sz, sz, sz)
        A = sine_trans(self.x4, s=sz)  # Operate on 6 arrays simultaneously
        return A[0], A[1], A[2], A[3], A[4], A[5]

    @profiler
    def sine_trans(self, sz=None):
        """Compute function values with type-I DST (identical in form as IDST).
        Return another LCState object containing function values."""
        if sz is None:
            sz = self.N
        x1 = LCState_s(sz)
        x1.x4[:] = sine_trans(self.x4, s=(sz, sz, sz))  # Operate on 6 arrays simultaneously
        return x1

    @profiler
    def phi_values(self, sz=None):
        if sz is None:
            sz = self.N
        return sine_trans(self.phi, s=(sz, sz, sz))

    @profiler
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

    @profiler
    def ydiff(self, aux_k2=None):
        """Compute value of y-derivatives"""
        N = self.N
        k2 = aux_k2 if aux_k2 is not None else \
            np.arange(1, N + 1).reshape([1, N, 1])
        xv = LCState_s(N)
        for i in range(6):
            xv.x4[i, :] = cos_trans(np.pi * k2 * self.x4[i, :], axis=1)
        return xv

    @profiler
    def zdiff(self, aux_k3=None):
        """Compute value of z-derivatives"""
        N = self.N
        k3 = aux_k3 if aux_k3 is not None else \
            np.arange(1, N + 1).reshape([1, 1, N])
        xv = LCState_s(N)
        for i in range(6):
            xv.x4[i, :] = cos_trans(np.pi * k3 * self.x4[i, :], axis=2)
        return xv

    @profiler
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


@profiler
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


@profiler
def view_as_lc(x0: np.ndarray, N):
    """Convert (or `view` in numpy terminology) arrays to LCState_s"""
    return LCState_s(N, x0.ravel())


@profiler
def save_lc(filename, X: LCState_s):
    """Save LC state (preserve size N for later retrieval)"""
    np.save(filename, X.x4)


@profiler
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
    # Test adjoint of the differentiation operator
    for i in range(10):
        X.x[:] = np.random.randn(6 * N**3)
        Y.x[:] = np.random.randn(6 * N**3)
        Xx = X.xdiff()
        print(np.sum(Xx.q1 * Y.q1),
              np.sum(X.q1 * np.pi * k1 * cos_trans(Y.q1, axis=0)))

    X1 = X.sine_trans(sz=31)
