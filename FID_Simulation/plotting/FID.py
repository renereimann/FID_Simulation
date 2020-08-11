import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as mcolors

from ..units import us, ms, uV

def plot_FID(times, fluxes, labels=None, inset=[[1.8, 2.8], [-1, 1]], savepath=None, close_on_exit=False):
    times = np.atleast_2d(times)
    fluxes = np.atleast_2d(fluxes)
    if labels is not None:
        labels = np.atleast_2d(labels)
    else:
        labels = [None]*len(fluxes)

    fig, ax = plt.subplots()
    for time, flux, label in zip(times, fluxes, labels):
        ax.plot(time/ms, flux/uV, label=label)
    ax.set_xlabel("t / ms")
    ax.set_xlim([0, 10])
    ax.set_ylabel("induced voltage in coil / $\mu$V")
    if any([l is not None for l in labels]):
        ax.legend()
    plt.tight_layout()

    if inset is not None:
        axins = inset_axes(ax, width="60%",  height="30%", loc=1)
        for time, flux in zip(times, fluxes):
            axins.plot(time/ms, flux/uV)
        axins.set_xlim(inset[0])
        axins.set_ylim(inset[1])
        axins.yaxis.get_major_locator().set_params(nbins=7)
        axins.xaxis.get_major_locator().set_params(nbins=7)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    if savepath is not None:
        fig.savefig(savepath.replace("png", "pdf"), bbox_inches="tight")
        fig.savefig(savepath.replace("pdf", "png"), dpi=200)
    if close_on_exit:
        plt.close()
