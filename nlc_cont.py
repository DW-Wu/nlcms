"""Test of parameters by continuation method while keeping record of as much
information as possible."""
from nlc_solve import *
from scipy.interpolate import interp1d

parser = ArgumentParser(prog="nlc_cont",
                        description="Solve the model with continuation method")
parser.add_argument('c0', action='store',
                    help="Config 0 (starting point)")
parser.add_argument('c1', action='store',
                    help="Config 1 (ending point)")
parser.add_argument('-o', '--output', action="store", default="out",
                    help="Output directory")
parser.add_argument('-i', '--init', action="store", default=None,
                    help="Initial value in .npy file")
parser.add_argument('-f', '--force-overwrite', action="store_true", default=False,
                    help="Force overwrite existing directory")
parser.add_argument('-s', '--silent', action="store_true", default=False,
                    help="Silent run (no printed messages)")
parser.add_argument('-N', '--num-sines', action="store", default=47, type=int,
                    help="Number of sine functions along each axis")
parser.add_argument('--maxiter', action='store', default=2000, type=int,
                    help="Maximum iteration number")
parser.add_argument('--eta', action='store', default=1e-4, type=float,
                    help="Gradient descent step length (learning rate)")
parser.add_argument('--tol', action='store', default=1e-8, type=float,
                    help="Gradient norm tolerance (relative to dimension)")


def _arr_to_conf(x):
    return LCConfig(**{k: v for (k, v) in zip(LCConfig.KEYS, x)})


def _conf_to_arr(c):
    return np.array([c[k] for k in LCConfig.KEYS])


def _interp_config(c0: LCConfig, c1: LCConfig, nsteps):
    """Interpolate between two LC configurations (containing endpoints)"""
    a0 = _conf_to_arr(c0)
    a1 = _conf_to_arr(c1)
    a1[[1, 2, 3, 5]] = a0[[1, 2, 3, 5]]  # Ensure B,C,L,v0 are identical
    inter = interp1d(x=[0, 1], y=[a0, a1], axis=0)
    return inter(np.arange(nsteps + 1) / nsteps)  # shape=(nsteps+1, 10)


def _refine_conf_range(c_arr, i, j, n):
    """Refine steps between indices i and j into n steps"""
    assert i < j, IndexError("Invalid index range")
    new_range = _interp_config(_arr_to_conf(c_arr[i, :]),
                               _arr_to_conf(c_arr[j, :]), n)
    return np.vstack([c_arr[:i], new_range, c_arr[j + 1:]])


class ContTest(LCSolve):
    def __init__(self, c0: LCConfig, c1: LCConfig, nsteps: int,
            N: int,
            x0=None,
            outdir="./out",
            force=True,
            verbose=False):
        if not force and exists(outdir):
            # Try to read existing config and state
            if verbose:
                print("Reading existing config and state")
            x0 = load_lc(join(outdir, "init.npy"), resize=N)
            self.c_arr = np.load(join(outdir, "conf.npy"))
            c0 = _arr_to_conf(self.c_arr[0])
        else:
            self.c_arr = _interp_config(c0, c1, nsteps)
        # Call initialization method of superclass
        # Makes directory if not exist
        # Initialize state X
        super().__init__(outdir, c0, N, x0, load_file=False, verbose=verbose)

        # Rewrite files
        self.sw = 2
        self.Evec = []  # Energy values in steps
        self.iternum = []  # Numbers of iteration (large values indicate borderline case)
        self.reset_clist()  # Save and export config files
        save_lc(join(self.outdir, "init.npy"), self.X)  # initial (from LCSolve.__init__)

    def reset_clist(self, c_arr=None):
        """Reset configuration list"""
        if c_arr is not None:
            self.c_arr = c_arr
        self.sw = int(np.floor(np.log10(self.c_arr.shape[0] - 1))) + 1
        self.Evec = np.zeros(self.c_arr.shape[0])
        self.iternum = np.zeros(self.c_arr.shape[0], dtype=int)
        np.save(join(self.outdir, "conf.npy"), self.c_arr)

    def refine_clist(self, ind=None):
        """Adaptively refine config range near the borderline"""
        nsteps = self.c_arr.shape[0]
        if ind is None:
            i = np.argmax(self.iternum)
            j = nsteps - 1 - np.argmax(self.iternum[::-1])  # argmax from the right
            if i > 0:
                i -= 1
            if j < nsteps - 1:
                j += 1
        else:
            i, j = ind
            assert 0 <= i and i < j and j < nsteps
        fine_c_arr = _refine_conf_range(self.c_arr, i, j, 2 * (j - i))
        # Rename state files to their new indices
        for k in range(j, nsteps):
            if exists(self._get_fname(k) + '.json'):
                os.rename(self._get_fname(k) + '.json',
                          self._get_fname(k + (j - i)) + '.json')
            if exists(self._get_fname(k) + '.npy'):
                os.rename(self._get_fname(k) + '.npy',
                          self._get_fname(k + (j - i)) + '.npy')
        self.reset_clist(fine_c_arr)

    def _get_fname(self, i):
        """Get output file name (index formatted into strings of uniform length)
        Example: if 0<nsteps<10 then"""
        return join(self.outdir, f"s{i:0>{self.sw}}")

    def main(self, disp_key=None, method="gd", **kwargs):
        if disp_key is not None:
            # Display the value of given parameter during continuation
            assert disp_key in LCConfig.KEYS, "Invalid parameter name"
        for i in range(self.c_arr.shape[0]):
            self.conf = _arr_to_conf(self.c_arr[i])
            save_lc_config(self._get_fname(i) + ".json", self.conf)
            # Load solution if possible
            if exists(self._get_fname(i) + '.npy'):
                self.X = load_lc(self._get_fname(i) + '.npy')
            # Call solver subroutine
            itn = self.solve(method=method, **kwargs)
            self.iternum[i] = itn
            self.Evec[i] = self.fvec[-1]  # energy value has been recorded
            # Display iteration info
            if disp_key is not None and kwargs.get('verbose', True):
                print("Step %d: %s = %.3e\t E = %.6f" %
                      (i, disp_key, self.conf[disp_key], self.Evec[i]))
            save_lc(self._get_fname(i) + ".npy", self.X)

    def load_state(self, i):
        return load_lc(self._get_fname(i) + ".npy")


if __name__ == "__main__":
    args = parser.parse_args()
    c0 = load_lc_config(args.c0)
    c1 = load_lc_config(args.c1)
    N = args.num_sines  # default 41
    if args.init is not None:
        x0 = load_lc(args.init, resize=N)
    else:
        x0 = None
    test = ContTest(c0, c1, 20,
                    N=N, x0=x0,
                    outdir=args.output,
                    force=args.force_overwrite,
                    verbose=not args.silent)
    test.refine_clist(ind=(19, 20))
    test.main(disp_key="lam", method="gd",
              maxiter=args.maxiter,  # default 2000
              eta=args.eta,  # default 1e-4
              tol=args.tol * np.sqrt(6 * N**3),  # default 1e-6
              verbose=not args.silent,
              bb=True)
