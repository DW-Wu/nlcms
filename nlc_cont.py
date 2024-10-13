"""Test of parameters by continuation method while keeping record of as many
information as possible."""
from nlc_solve import *
from shutil import rmtree


class ContTest:
    def __init__(self, c0: LCConfig, c1: LCConfig, nsteps: int, x0=None, **kwargs):
        # Output directory
        self.outdir = kwargs.get("outdir", "./out")
        # Default state (if None, reconstruct a sphere)
        if x0 is None:
            x0 = LCState_s(31)
            xx = np.arange(1, 32) / 32
            xx, yy, zz = np.meshgrid(xx, xx, xx, indexing='ij')
            r = np.sqrt((xx - .5)**2 + (yy - .5)**2 + (zz - .5)**2)
            phiv = (np.tanh(((3 * c0.v0 / 4 / np.pi)**(1 / 3) - r) / 0.04) + 1)
            x0.phi[:] = 4. * fft.idstn(phiv, type=1)
        x0.proj_phi(c0.v0)
        # Default configuration
        self.metadata = {"c0": dict(c0),
                         "c1": dict(c1),
                         "nsteps": nsteps,
                         "N": x0.N,
                         "method": 'gd',  # algorithm (to be implemented)
                         "eta": 1e-3,  # it. step size
                         "maxiter": 1000,  # it. in each continuation step
                         "tol": 1e-8}
        self.metadata.update(kwargs)
        force = kwargs.get("force", False)
        if not exists(self.outdir) or force:
            # Initialize new directory
            # or force create new directory (by overwriting)
            if exists(self.outdir):
                if force:
                    print("Force deleting existing directory")
                    rmtree(self.outdir)
                    os.makedirs(self.outdir)
                else:
                    print("Overwriting existing directory")
            else:
                os.makedirs(self.outdir)
            save_lc(join(self.outdir, "init.npy"), x0)
            self.X = x0  # Store state
            with open(join(self.outdir, "config.json"), mode='w', encoding='utf-8') as cf:
                json.dump(self.metadata, cf,
                          separators=(',', ': '), indent=2)  # store config file
        if exists(self.outdir) and not force:
            print("Trying to read from existing directory")
            self.X = load_lc(join(self.outdir, "init.npy"))
            with open(join(self.outdir, "config.json"), mode='r') as cf:
                self.metadata.update(json.load(cf))
            self.metadata['N'] = self.X.N  # in case N is inconsistent in config and state
            with open(join(self.outdir, "config.json"), mode='w') as cf:
                json.dump(self.metadata, cf,
                          separators=(',', ': '), indent=2)  # Export again

        # Get basic info from metadata (provided by program or read from )
        c0 = LCConfig(**self.metadata['c0'])
        c1 = LCConfig(**self.metadata['c1'])
        self.B = c0.B  # B,C,L,v0 are unchangeable
        self.C = c0.C
        self.L = c0.L
        self.v0 = c0.v0
        self.c0 = np.array([c0.A, c0.lam, c0.eps, c0.omega, c0.wp, c0.wv])  # store config as vector
        self.c1 = np.array([c1.A, c1.lam, c1.eps, c1.omega, c1.wp, c1.wv])
        self.nsteps = self.metadata['nsteps']
        self.sw = int(np.ceil(np.log10(self.nsteps)))  # Width in format string
        self.E_vec = []

    def _get_fname(self, i):
        """Get output file name (index formatted into strings of uniform length)"""
        return join(self.outdir, f"{i:0>{self.sw}}")

    def main(self, disp_key=None):
        if disp_key is not None:
            # Display the value of given parameter during continuation
            assert disp_key in ['A', 'B', 'C', 'L', 'lam', 'v0', 'eps', 'omega',
                                'wp', 'wv'], "Invalid parameter name"
        N = self.X.N
        FF = LCFunc_s()
        if exists(join(self.outdir, 'snapshot.npy')):
            self.X = load_lc(join(self.outdir, 'snapshot.npy'))
        for i in range(self.nsteps + 1):
            r = i / self.nsteps
            c_vec = (1 - r) * self.c0 + r * self.c1  # current config
            c = LCConfig(A=c_vec[0], B=self.B, C=self.C, L=self.L,
                         lam=c_vec[1], v0=self.v0, eps=c_vec[2],
                         omega=c_vec[3], wp=c_vec[4], wv=c_vec[5])
            FF.reset_conf(c)
            save_lc_config(self._get_fname(i) + ".json", c)
            # g=FF.grad(self.X,proj=True)
            # if norm(g.x)>0.1:
            #     # Use implicit gradient flow to smoothen
            #     self.X, _= solve_gf(FF, self.X, maxiter=200, dt=1e-4, tol=1e-8)
            # Then BB gradient descent
            if exists(self._get_fname(i) + '.npy'):
                self.X = load_lc(self._get_fname(i) + '.npy')  # use existing solution file
            if self.metadata['method'] == 'gd':
                self.X, _ = solve_gd(FF, self.X,
                                     maxiter=self.metadata['maxiter'],
                                     eta=self.metadata['eta'],
                                     tol=self.metadata['tol'] * np.sqrt(6 * N**3),
                                     bb=True,
                                     verbose=True,
                                     inspect=False)
            self.E_vec.append(FF.energy(self.X))
            if disp_key is not None:
                print("Step %d: %s = %.3e\t E = %.6f" %
                      (i, disp_key, c[disp_key], self.E_vec[-1]))
            save_lc(self._get_fname(i) + ".npy", self.X)

    def load_state(self, i):
        return load_lc(self._get_fname(i) + ".npy")

    def snapshot(self):
        """Save current state file in case of program break"""
        save_lc(join(self.outdir, 'snapshot.npy'), self.X)


if __name__ == "__main__":
    c0 = load_lc_config("ring.json")
    c1 = load_lc_config("tactoid.json")
    x0 = load_lc("ring48.npy")
    test = ContTest(c0, c1, 20, x0=x0, outdir='./out/test1',
                    force=False, method="gd", eta=1e-3, maxiter=2000, tol=1e-8)
    try:
        test.main(disp_key='lam')
    except:
        test.snapshot()  # save state file
