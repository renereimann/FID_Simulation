import numpy as np
from scipy.optimize import minimize
from scipy.ndimage.filters import uniform_filter1d
from ..units import *
from .hilbert_transform import HilbertTransform
import matplotlib.pyplot as plt

# optimal window size is 2pi/omega_bar
def running_mean(time, val, window_size):
    N = int(window_size/np.diff(time)[0])
    if N%2 == 0:
        N += 1
    val = uniform_filter1d(val, size=N)
    return time, val

fit_func = lambda t, p: p[0] + p[1]*t + p[2]*t**3 + p[3]*t**5 + p[4]*t**7

def fit_range(time, amp, edge_ignore=0.1*ms, frac=0.7,
              return_mask=False, pretrigger=None, readout_length=None):
    t_min = np.min(time)
    t_max = np.max(time)
    if pretrigger is not None:
        t_min = np.max([t_min, pretrigger])
    if readout_length is not None:
        t_max = np.min([t_max, readout_length])
    mask_edge = np.logical_and(time > t_min + edge_ignore,
                               time < t_max - edge_ignore)

    thres = np.max(amp[mask_edge])*frac
    mask = np.logical_and(mask_edge, amp > thres)

    t_min = np.min(time[mask])
    t_max = np.min(time[np.logical_and(time > t_min, np.logical_not(mask))])
    t_range = np.array([t_min, t_max])
    if return_mask:
        return t_range, np.logical_and(time > t_min, time < t_max)
    return t_range

def chi2_min_diagonal(time, phi, amp, time_range=[0,2], sigma_N=1):
    mask = np.logical_and(time > np.min(time_range), time < np.max(time_range))
    time = time[mask][::2] # downsample by factor 2
    phi = phi[mask][::2]   # downsample by factor 2
    amp = amp[mask][::2]   # downsample by factor 2
    sigma_inv = amp**2/sigma_n**2
    chi2 = lambda p: np.sum(sigma_inv*(fit_func(time, p) - phi)**2)

    x0 = np.random.normal(size=5)
    x0[1] += 314
    res = minimize(chi2, x0)
    return res

def poly_fit(time, phi,time_range=[0,2], x0=None, tol=1e-4):
    mask = np.logical_and(time > np.min(time_range), time < np.max(time_range))

    chi2 = lambda p: np.sum((fit_func(time[mask], p) - phi[mask])**2)
    if x0 is None:
        x0 = np.random.normal(size=5)
    res = minimize(chi2, x0, tol=tol)
    return res

def phase_analysis(time, flux, t0, t_range, verbose=False, plotting=False, smoothing=True, tol=None, x0=None, window_size=1/(314*kHz), fit_window_fact=2):
    hilbert = HilbertTransform(time, flux)
    _, env = hilbert.EnvelopeFunction()
    _, phi = hilbert.PhaseFunction()

    phi_raw = phi[:]
    if smoothing:
        _, phi = running_mean(time,  phi, window_size=window_size)

    width = (t_range[1]-t_range[0])/fit_window_fact
    mask = np.logical_and(time > t_range[0], time < t_range[1])
    res = poly_fit((time-t0)/width, phi, time_range=(t_range-t0)/width, x0=x0, tol=tol)
    phi_fit = fit_func((time[mask]-t0)/width, res.x)
    frequency = res.x[1]/width
    phi0 = res.x[0]

    if verbose:
        print("frequency:", frequency/kHz, "kHz")

    if plotting:
        plt.plot(time/ms, phi_raw - phi0 - frequency*(time-t0), color="b", label="raw FID")
        if smoothing:
            plt.plot(time/ms, phi - phi0 - frequency*(time-t0), color="red", label="smoothed FID")
        plt.plot(time[mask]/ms, phi_fit - phi0 - frequency*(time[mask]-t0), color="k", ls="--", label="fit")
        plt.grid()
        plt.axvspan(*(t_range/ms), color="gray", alpha=0.2)
        plt.ylim(-0.1, 0.2)
        plt.xlabel("time / ms")
        plt.xlim( (np.mean(t_range)-0.55*(t_range[1]-t_range[0]))/ms, (np.mean(t_range)+0.55*(t_range[1]-t_range[0]))/ms)
        plt.ylabel(r"$ \Phi(t) - \hat{\Phi_0} - \hat{\frac{\mathrm{d}\Phi}{\mathrm{d}t}}\cdot t$")
        plt.legend()
    return frequency

def FID_analysis(time, flux, edge_ignore=60*us, frac=0.7, probe=None, **kwargs):
    t0 = probe.time_pretrigger
    _, env = HilbertTransform(time, flux).EnvelopeFunction()
    t_range = fit_range(time, env,
                        frac=frac, edge_ignore=edge_ignore,
                        pretrigger=probe.time_pretrigger,
                        readout_length=probe.readout_length)
    frequency = phase_analysis(time, flux, t0, t_range, **kwargs)

    if "plotting" in kwargs.keys() and kwargs["plotting"]:
        plt.axvline(t0/ms, ls="--", color="k")
        plt.text(t0/ms, 0.20, "trigger", rotation=90, va="top", ha="left", fontsize=12, fontweight='bold')
        plt.xlim(xmin=t0/ms*0.95)
    return frequency

def Echo_analysis(time, flux, t_window=0.5*ms, probe=None, **kwargs):
    t0 = 2*probe.readout_length-probe.time_pretrigger
    t_range = np.array([t0-t_window, t0+t_window])
    frequency = phase_analysis(time, flux, t0, t_range, **kwargs)

    if "plotting" in kwargs.keys() and kwargs["plotting"]:
        plt.axvline(t0/ms, ls="--", color="k")
        plt.text(t0/ms, 0.20, "Echo", rotation=90, va="top", ha="left", fontsize=12, fontweight='bold')

    return frequency
