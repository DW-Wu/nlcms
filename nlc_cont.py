"""Test of parameters by continuation method while keeping record of as much
information as possible."""
from nlc_solve import *
from scipy.interpolate import interp1d
from shutil import move

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
parser.add_argument('--refine-steps', action='store', default=3, type=int,
                    help="Steps of auto refinement")
parser.add_argument('--num-steps', action='store', default=20, type=int,
                    help="Initial number of steps between endpoint configs")


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
        if verbose:
            print("Starting config:", self.c_arr[0])
            print("End config:", self.c_arr[-1])
            print("Number of steps:", self.c_arr.shape[0])
            print("Output path:", os.path.abspath(self.outdir))

        # Rewrite files
        self.sw = 0
        self.Evec = []  # Energy values in steps
        self.iternum = []  # Numbers of iteration (large values indicate borderline case)
        self._set_clist()  # Save and export config files
        save_lc(join(self.outdir, "init.npy"), self.X)  # initial (from LCSolve.__init__)

    def _set_clist(self, c_arr=None):
        """Reset configuration list"""
        if c_arr is not None:
            self.c_arr = c_arr
        old_sw = self.sw
        self.sw = int(np.floor(np.log10(self.c_arr.shape[0] - 1))) + 1
        if self.sw != old_sw:
            # Reformat filenames
            for fn in os.listdir(self.outdir):
                if fn.startswith('s') and fn[1].isnumeric():
                    suffix = fn[fn.find('.'):]
                    num = int(fn[1:fn.find('.')])
                    os.rename(join(self.outdir, fn),
                              self._get_fname(num) + suffix)
        self.Evec = np.zeros(self.c_arr.shape[0])
        self.iternum = np.zeros(self.c_arr.shape[0], dtype=int)
        np.save(join(self.outdir, "conf.npy"), self.c_arr)

    def _refine_steps(self, i, j):
        """Refine the step between indices i and j"""
        nsteps = self.c_arr.shape[0]
        assert 0 <= i and i < j and j < nsteps
        # Halve the step size
        new_range = _interp_config(_arr_to_conf(self.c_arr[i, :]),
                                   _arr_to_conf(self.c_arr[j, :]), 2 * (j - i))
        fine_c_arr = np.vstack([self.c_arr[:i],
                                new_range,
                                self.c_arr[j + 1:]])
        old_iternum = self.iternum
        self._set_clist(fine_c_arr)
        # Rename state files to their new indices
        # Start from largest index to avoid filename conflict
        for k in range(nsteps - 1, j, -1):
            if exists(self._get_fname(k) + '.json'):
                move(self._get_fname(k) + '.json',
                     self._get_fname(k + (j - i)) + '.json')
            if exists(self._get_fname(k) + '.npy'):
                move(self._get_fname(k) + '.npy',
                     self._get_fname(k + (j - i)) + '.npy')
        # Retrieve old iteration numbers, where step length did not change
        self.iternum[:i + 1] = old_iternum[:i + 1]
        # self.iternum[2 * j - i + 1:] = old_iternum[j + 1:]

    def auto_refine(self, ind=None):
        """Adaptively auto refine config range near phase changes, i.e. large
        iteration numbers"""
        nsteps = self.c_arr.shape[0]
        maxit = max(self.iternum)
        ind = []
        i = 0
        # Find all continuous ranges of maximum iteration numbers
        while i < nsteps:
            if self.iternum[i] < maxit:
                i += 1  # Find beginning of range
            else:
                j = i
                while j < nsteps and self.iternum[j] == maxit:
                    j += 1  # Find end of range
                ind.append((max(0, i - 1), min(j, nsteps - 1)))
                i = j
        while ind:
            # Start interpolation from the back
            self._refine_steps(*ind.pop())

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
            # Load solution if the iteration has been done before
            if (i == 0 or self.iternum[i] > 0) and exists(self._get_fname(i) + '.npy'):
                # Ensure consistent size when loading initial file
                self.X = load_lc(self._get_fname(i) + '.npy', resize=self.X.N)
                # Call solver subroutine
                self.solve(method=method, **kwargs)
            else:
                # Save new iternum
                self.iternum[i] = self.solve(method=method, **kwargs)
            self.Evec[i] = self.fvec[-1]  # energy value has been recorded
            # Display iteration info
            if disp_key is not None and kwargs.get('verbose', True):
                print("Step %d: %s = %.3e\t E = %.6f" %
                      (i, disp_key, self.conf[disp_key], self.Evec[i]))
            save_lc(self._get_fname(i) + ".npy", self.X)

    def load_state(self, i):
        return load_lc(self._get_fname(i) + ".npy")

    def snapshot(self, fname="snapshot.json"):
        """Write runtime information into file"""
        D = {"length": self.c_arr.shape[0], "iternum": list(self.iternum)}
        with open(join(self.outdir, fname), mode='w') as f:
            json.dump(D, f)


if __name__ == "__main__":
    args = parser.parse_args()
    c0 = load_lc_config(args.c0)
    c1 = load_lc_config(args.c1)
    N = args.num_sines  # default 47
    if args.init is not None:
        x0 = load_lc(args.init, resize=N)
    else:
        x0 = None
    test = ContTest(c0, c1, args.num_steps,  # default 20
                    N=N, x0=x0,
                    outdir=args.output,
                    force=args.force_overwrite,
                    verbose=not args.silent)
    test.main(disp_key="lam", method="gd",
              maxiter=args.maxiter,  # default 2000
              eta=args.eta,  # default 1e-4
              tol=args.tol * np.sqrt(6 * N**3),  # default 1e-6
              verbose=not args.silent,
              bb=True)
    for _ in range(args.refine_steps):
        # 3 rounds of adaptive refinement
        test.auto_refine()
        test.main(disp_key="lam", method="gd",
                  maxiter=args.maxiter,  # default 2000
                  eta=args.eta,  # default 1e-4
                  tol=args.tol * np.sqrt(6 * N**3),  # default 1e-6
                  verbose=not args.silent,
                  bb=True)
