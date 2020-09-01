import numpy as np
from scipy.optimize import minimize
from scipy.ndimage.filters import uniform_filter1d
from ..units import *

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
