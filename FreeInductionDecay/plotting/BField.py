import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ..units import mm, ppm, T

def plot_cells_bfield_3d(probe, title=None, cmap="seismic",
                          close_on_exit=False, savepath=None,
                          ref_B0=1.45*T, outline_probe=True):

    colors = (probe.cells_B0-ref_B0)/ref_B0/ppm
    dB = np.max(np.abs(colors))

    fig = plt.figure()
    gs = GridSpec(5, 6)
    ax1 = plt.subplot(gs[0:2, :])
    ax2 = plt.subplot(gs[2:, :3])
    ax3 = plt.subplot(gs[2:, 3:], projection='3d')

    if title is not None:
        fig.suptitle(title)

    ax1.scatter(probe.cells_z/mm,
                probe.cells_y/mm,
                c=colors, cmap=cmap, vmin=-dB, vmax=dB, s=3)
    ax1.set_aspect("equal")
    ax1.set_xlabel("z/mm")
    ax1.set_ylabel("y/mm")
    ax1.set_xlim(-15, 15)

    sc = ax2.scatter(probe.cells_x/mm,
                     probe.cells_y/mm,
                     c=colors, cmap=cmap, vmin=-dB, vmax=dB, s=3)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x/mm")
    ax2.set_ylabel("y/mm")
    plt.colorbar(sc, ax=ax2).set_label(r"$\Delta B\, /\, \mathrm{ppm}$")

    sc = ax3.scatter(probe.cells_x/mm,
                     probe.cells_z/mm,
                     probe.cells_y/mm,
                     c=colors, cmap=cmap, vmin=-dB, vmax=dB, s=3)
    ax3.set_xlim(-5,5)
    ax3.set_ylim(-5,5)
    ax3.set_zlim(-5,5)
    ax3.set_xlabel("x/mm")
    ax3.set_ylabel("z/mm")
    ax3.set_zlabel("y/mm")

    if outline_probe:
        # Plot the cylinder
        u = np.linspace(0, 2 * np.pi, 8)
        x = probe.radius/mm * np.outer(np.cos(u), np.ones(8))
        y = np.outer(np.ones(8), np.linspace(-probe.length/mm/2, probe.length/mm/2, 8))
        z = probe.radius/mm * np.outer(np.sin(u), np.ones(8))
        ax3.plot_wireframe(x, y, z, color='k', alpha=0.4)
        #ax.plot_surface(x, y, z, color='b', alpha=0.3)

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath.replace("pdf", "png"), dpi=200)
        fig.savefig(savepath.replace("png", "pdf"), bbox_inches="tight")
    if close_on_exit:
        plt.close()

def hist_cells_x_y(probe, cmap="gist_yarg", savepath=None, close_on_exit=False):
    fig, ax = plt.subplots()
    sc = ax.hist2d(probe.cells_x/mm,
                   probe.cells_y/mm,
                   bins=[np.linspace(-probe.radius/mm, probe.radius/mm, 33),
                         np.linspace(-probe.radius/mm, probe.radius/mm, 33)],
                   cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel("x/mm")
    ax.set_ylabel("y/mm")
    plt.colorbar(sc[-1]).set_label("# Cells")

    fig.tight_layout()
    if savepath is not None:
        plt.savefig(savepath.replace("pdf", "png"), dpi=200)
        plt.savefig(savepath.replace("png", "pdf"), bbox_inches="tight")
    if close_on_exit:
        plt.close()

def hist_cells_z_y(probe, cmap="gist_yarg", savepath=None, close_on_exit=False):
    fig, ax = plt.subplots()
    sc = ax.hist2d(probe.cells_z/mm,
                   probe.cells_y/mm,
                   bins=[np.linspace(-probe.length/mm/2, probe.length/mm/2, 641),
                         np.linspace(-probe.radius/mm, probe.radius/mm, 33)],
                   cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel("z/mm")
    ax.set_ylabel("y/mm")
    plt.colorbar(sc[-1], orientation="horizontal", shrink=0.5).set_label("# Cells")

    fig.tight_layout()
    if savepath is not None:
        plt.savefig(savepath.replace("pdf", "png"), dpi=200)
        plt.savefig(savepath.replace("png", "pdf"), bbox_inches="tight")
    if close_on_exit:
        plt.close()

def plot_dB_vs_y(probe, ref_B0=1.45*T, cmap="seismic",
                 savepath=None, close_on_exit=False):
    fig, ax = plt.subplots()
    sc = ax.scatter(probe.cells_y/mm, (probe.cells_B0-ref_B0)/(ref_B0)/ppm,
            c=(probe.cells_z)/mm,
            cmap="seismic", vmin=-probe.length/mm/2, vmax=probe.length/mm/2)
    plt.colorbar(sc, ax=ax).set_label("z/mm")
    ax.set_xlabel("y/mm")
    ax.set_ylabel(r"$\Delta |B_\mathrm{ext}| / ppm$")

    fig.tight_layout()
    if savepath is not None:
        plt.savefig(savepath.replace("pdf", "png"), dpi=200)
        plt.savefig(savepath.replace("png", "pdf"), bbox_inches="tight")
    if close_on_exit:
        plt.close()
