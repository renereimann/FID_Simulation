from E989 import StorageRingMagnet, FixedProbe
from noise import FreqNoise
from units import *

import time
import numpy as np
from scipy import integrate
from scipy import fftpack

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as mcolors
from FFT_analysis import windowFunction

external_field = StorageRingMagnet( )
external_field.An[8] = 5e-6*external_field.An[2]/cm
nmr_probe = FixedProbe(B_field=external_field)
impedance = 50 * ohm
guete = 1.1
pulse_power = 10*W
nmr_probe.coil.current = guete * np.sqrt(2*pulse_power/impedance)

noise = FreqNoise(power=-1, scale=3*uV)
################################################################################
# calculation

# apply RF field

nmr_probe.apply_rf_field()


if True:
    fig, ax = plt.subplots(ncols=2)
    ax[0].scatter(nmr_probe.cells_x/mm, nmr_probe.cells_y/mm, c=(nmr_probe.cells_B0-1.45*T)/T, cmap="jet")
    ax[0].set_aspect("equal")
    ax[1].scatter(nmr_probe.cells_z/mm, nmr_probe.cells_y/mm, c=(nmr_probe.cells_B0-1.45*T)/T, cmap="jet")
    ax[1].set_aspect("equal")


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = (nmr_probe.cells_B0-1.45*T)/T
    my_col = mcolors.seismic((X-np.amin(X))/(np.amax(X)-np.amin(X)))
    ax.scatter(nmr_probe.cells_x/mm, nmr_probe.cells_y/mm, nmr_probe.cells_z/mm, c=my_col, marker="o")
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    ax.set_zlim(-20,20)
    plt.show()

# calculate FID
# trolly: 1 MSPS
# fix probes: 10 MSPS --> totally oversampled
times = np.linspace(0*ms, 10*ms, 10000) # 1 MSPS
t_start = time.time()
flux = nmr_probe.pickup_flux(times, mix_down=61.74*MHz)
print("Needed", time.time()-t_start, "sec to calculate FID.")

flux_noise = noise(times)
flux += flux_noise

N = len(times)                # Number of samplepoints
windowFunction = {"Hann": lambda nN: (np.sin(np.pi*nN))**2,
                  "Rectangular": lambda nN: 1,
                  "Triangular": lambda nN: 1-np.abs(2*nN - 1),
                  "Welch": lambda nN: 1 - (2*nN-1)**2,
                  "sine": lambda nN: np.sin(np.pi*nN),
                  "Blackman": lambda nN: 7938/18608 - 9240/18608 * np.cos(2*np.pi*nN) + 1430/18608*np.cos(4*np.pi*nN),
                  "Nuttal": lambda nN: 0.355768-0.487396*np.cos(2*np.pi*nN) +0.144232*np.cos(4*np.pi*nN) - 0.012604 *np.cos(6*np.pi*nN),
                  "Blackman-Nuttal": lambda nN: 0.3635819-0.4891775*np.cos(2*np.pi*nN) +0.1365995*np.cos(4*np.pi*nN) - 0.0106411 *np.cos(6*np.pi*nN),
                  }
yf = fftpack.fft(flux*windowFunction["Hann"](np.linspace(0,1,len(flux))))
xf = np.linspace(0.0, 1.0/(2.0*(times[1]-times[0]))/kHz, N/2)
################################################################################
# plotting

if False:
    # make a plot for comparison
    cross_check = np.genfromtxt("./RF_coil_field_cross_check.txt", delimiter=", ")
    zs = np.linspace(-15*mm, 15*mm, 1000)
    plt.figure()
    plt.plot(cross_check[:,0], cross_check[:,1], label="Cross-Check from DocDB 16856, Slide 5\n$\O=2.3\,\mathrm{mm}$, $L=15\,\mathrm{mm}$, turns=30", color="orange")
    B_rf_z_0 = nmr_coil.B_field(0, 0, 0)[2]
    plt.plot(zs/mm, [nmr_coil.B_field(0, 0, z)[2] / B_rf_z_0 for z in zs], label="My calculation\n$\O=4.6\,\mathrm{mm}$, $L=15\,\mathrm{mm}$, turns=30", color="k", ls=":")
    plt.xlabel("z / mm")
    plt.ylabel("$B_z(0,0,z)\, /\, B_z(0,0,0)$")
    plt.legend(loc="lower right")
    plt.title("Magnetic field of the coil (static)")
    plt.tight_layout()
    plt.savefig("./plots/coil_field_distribution.pdf", bbox_inches="tight")

if False:
    zs = np.linspace(-15*mm, 15*mm, 1000)
    cross_check = np.genfromtxt("./mu_y_vs_z.txt", delimiter=" ")
    plt.figure()
    plt.plot(cross_check[:,0], cross_check[:,1], label="$\mu_y$, Cross-Check from DocDB 16856, Slide 7", color="orange")
    #scan = np.array([mu_x(0*mm,0*mm,z) for z in zs])
    plt.scatter(nmr_probe.cells_z/mm, nmr_probe.cells_mu_x, label="$\mu_x$")
    plt.scatter(nmr_probe.cells_z/mm, nmr_probe.cells_mu_y, label="$\mu_y$")
    plt.scatter(nmr_probe.cells_z/mm, nmr_probe.cells_mu_z, label="$\mu_z$")
    plt.xlabel("z / mm")
    plt.ylabel("see legend")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/magnitization_after_pi2_pulse.pdf", bbox_inches="tight")
    plt.show()

if True:
    fig, ax = plt.subplots()
    ax.plot(times/ms, flux/uV)
    ax.plot(times/ms, flux_noise/uV)
    ax.plot(times/ms, (np.sin(np.pi*np.linspace(0,1,len(flux))))**2)
    ax.set_xlabel("t / ms")
    ax.set_xlim([0, 10])
    ax.set_ylabel("induced voltage in coil / $\mu$V")
    plt.tight_layout()
    axins = inset_axes(ax, width="60%",  height="30%", loc=1)
    axins.plot(times/ms, flux/uV)
    axins.plot(times/ms, flux_noise/uV)
    axins.set_xlim([1.8, 2.8])
    axins.set_ylim([-1, 1])
    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    fig.savefig("./plots/FID_signal.pdf", bbox_inches="tight")


    fig = plt.figure()
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.xlabel("f / kHz")
    plt.xlim([0, 1.0/(2.0*(times[1]-times[0]))/kHz])
    plt.ylabel("|FFT(f)|")
    plt.tight_layout()


    fig = plt.figure()
    mix_down = 61.74*MHz
    plt.hist((nmr_probe.material.gyromagnetic_ratio*nmr_probe.cells_B0 -2*np.pi*mix_down)/(2*np.pi*kHz))
    plt.show()

if True:
    freq = np.fft.fftfreq(N, d=times[1]-times[0])
    hilbert = fftpack.ifft(complex(0,-1)*np.sign(freq)*yf)
    phi = np.arctan(hilbert/yf)
    print(phi)
    plt.figure()
    plt.plot(times/ms, phi)
    plt.show()
