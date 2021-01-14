# -*- coding: utf-8 -*-
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as mcolors

from ..units import us

def plot_coil_bz():
    # make a plot for comparison
    fig, ax = plt.subplots()

    B_rf_z_0 = nmr_probe.coil.B_field(0, 0, 0)[2]
    ax.plot(zs/mm, [nmr_probe.coil.B_field(0, 0, z)[2] / B_rf_z_0 for z in zs], label="My calculation\n$\O=4.6\,\mathrm{mm}$, $L=15\,\mathrm{mm}$, turns=30", color="k", ls=":")

    ax.set_xlabel("z / mm")
    ax.set_ylabel("$B_z(0,0,z)\, /\, B_z(0,0,0)$")
    plt.legend(loc="lower right")
    plt.title("Magnetic field of the coil (static)")
    fig.tight_layout()

    if show_cross_check:
        cross_check = np.genfromtxt("../tests/RF_coil_field_cross_check.txt", delimiter=", ")
        zs = np.linspace(-15*mm, 15*mm, 1000)
        ax.plot(cross_check[:,0], cross_check[:,1], label="Cross-Check from DocDB 16856, Slide 5\n$\O=2.3\,\mathrm{mm}$, $L=15\,\mathrm{mm}$, turns=30", color="orange")
    if show_analytic_solution:
        ax.scatter(zs/mm, B_z(zs)/B_z(0), label="Experimentalphysik 2, Demtröder\nSection 3.2.6.d Page 95")

    if savepath is not None:
        fig.savefig(savepath.replace("png", "pdf"), bbox_inches="tight")
        fig.savefig(savepath.replace("pdf", "png"), dpi=200)
    if close_on_exit:
        plt.close()

def plot_RF_pulse_3D(data, savepath=None, close_on_exit=False):
    fig = plt.figure(figsize=(16,12))

    ax = fig.add_subplot(221)
    ax.set_title(r"$x \approx y \approx z \approx 0$")
    ax.plot(data["time"]/us, data["Mx_center"], label="$M_x$")
    ax.plot(data["time"]/us, data["Mz_center"], label="$M_z$")
    ax.plot(data["time"]/us, data["My_center"], label="$M_y$", color="k")
    ax.legend(loc=3)
    ax.grid()
    ax.set_xlim(0,14)
    ax.set_ylim(-1,1)
    ax.set_xlabel("time / $\mu$s")
    ax.set_ylabel(r"$M(\sim0,\sim0,\sim0)$")
    axins = inset_axes(ax, width="50%",  height="30%", loc=1)
    axins.plot(data["time"]/us, data["Mx_center"])
    axins.plot(data["time"]/us, data["Mz_center"])
    axins.plot(data["time"]/us, data["My_center"], color="k")
    axins.set_xlim(0.5,0.7)
    axins.set_ylim(-0.25,0.25)
    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    plt.yticks(visible=False)
    #mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax = fig.add_subplot(222)
    ax.set_title("averaged over volume")
    ax.plot(data["time"]/us, data["Mx_mean"], label="$M_x$")
    ax.plot(data["time"]/us, data["Mz_mean"], label="$M_z$")
    ax.plot(data["time"]/us, data["My_mean"], label="$M_y$", color="k")
    ax.legend(loc=3)
    ax.grid()
    ax.set_xlim(0,14)
    ax.set_ylim(-1,1)
    ax.set_xlabel("time / $\mu$s")
    ax.set_ylabel(r"$\langle M_i \rangle$")
    axins = inset_axes(ax, width="50%",  height="30%", loc=1)
    axins.plot(data["time"]/us, data["Mx_mean"])
    axins.plot(data["time"]/us, data["Mz_mean"])
    axins.plot(data["time"]/us, data["My_mean"], color="k")
    axins.set_xlim(0.5,0.7)
    axins.set_ylim(-0.25,0.25)
    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    plt.yticks(visible=False)
    #mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    step = -10
    ax = fig.add_subplot(223, projection='3d')
    sc = ax.scatter(data["Mx_center"][::step], data["Mz_center"][::step], data["My_center"][::step], c=data["time"][::step]/us,
                    marker="o", cmap="jet", vmin=data["time"][0]/us, vmax=data["time"][-1]/us)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel("Mx")
    ax.set_ylabel("Mz")
    ax.set_zlabel("My")

    ax.xaxis.get_major_locator().set_params(nbins=5)
    ax.yaxis.get_major_locator().set_params(nbins=5)
    ax.zaxis.get_major_locator().set_params(nbins=5)
    plt.colorbar(sc, ax=ax).set_label("time / $\mu$s")


    ax = fig.add_subplot(224, projection='3d')
    sc = ax.scatter(data["Mx_mean"][::step], data["Mz_mean"][::step], data["My_mean"][::step], c=data["time"][::step]/us,
                    #sc = ax.scatter(data[:,1], data[:,3], -data[:,2], c=data[::-1,0]/us,
                    marker="o", cmap="jet", vmin=data["time"][0]/us, vmax=data["time"][-1]/us)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    ax.set_xlabel("Mx")
    ax.set_ylabel("Mz")
    ax.set_zlabel("My")

    ax.xaxis.get_major_locator().set_params(nbins=5)
    ax.yaxis.get_major_locator().set_params(nbins=5)
    ax.zaxis.get_major_locator().set_params(nbins=5)
    plt.colorbar(sc, ax=ax).set_label("time / $\mu$s")

    if savepath is not None:
        fig.savefig(savepath.replace("png", "pdf"), bbox_inches="tight")
        fig.savefig(savepath.replace("pdf", "png"), dpi=200)
    if close_on_exit:
        plt.close()
