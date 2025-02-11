"""Plot split-core phase transition"""
import sys
from os.path import join
from nlc_func import *
from nlc_cont import _arr_to_conf
import matplotlib.pyplot as plt


def plot_energy(ax, dirname, name, pi):
    F = LCFunc_s()
    cc = np.load(join(dirname, "conf.npy"))
    E_vec = []
    for i in range(cc.shape[0]):
        c = _arr_to_conf(cc[i])
        F.reset_conf(c)
        x = load_lc(join(dirname, "s%02d.npy" % i))
        E_vec.append(F.energy(x))
    return ax.plot(cc[:, pi], E_vec, label=name)


if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 3.6))
    
    ax=fig.add_subplot(1,2,1)
    plot_energy(ax, "out/rd2rn", "radial", 0)
    plot_energy(ax, "out/rn2rd", "ring", 0)
    ax.set(title="Adjusting $A$", xlabel="$A$", ylabel="Energy")
    ax.legend()

    ax=fig.add_subplot(1,2,2)
    plot_energy(ax, "out/rd2tc", "radial", 4)
    plot_energy(ax, "out/tc2rd", "tactoid", 4)
    ax.set(title="Adjusting $\\lambda$", xlabel="$\\lambda$", ylabel="Energy")
    ax.legend()

    plt.savefig("out/energy_plot.pdf")
