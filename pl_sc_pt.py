"""Plot split-core phase transition"""
from os.path import join
from nlc_func import *
from nlc_cont import _arr_to_conf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    for dir, name in zip(["out/sc2tc", "out/tc2w"],
                        ["split-core", "tactoid"]):
        F = LCFunc_s()
        cc = np.load(join(dir, "conf.npy"))
        E_vec = []
        for i in range(cc.shape[0]):
            c = _arr_to_conf(cc[i])
            F.reset_conf(c)
            x = load_lc(join(dir, "s%02d.npy" % i))
            E_vec.append(F.energy(x))
        plt.plot(cc[:, 4], E_vec, "o--", label=name)
    plt.legend()
    plt.savefig("out/energy_plot.pdf")
