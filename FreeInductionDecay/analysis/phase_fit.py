import numpy as np
from scipy.optimize import minimize
from scipy.ndimage.filters import uniform_filter1d

# optimal window size is 2pi/omega_bar
def running_mean(time, val, window_size):
    N = int(window_size/np.diff(time)[0])
    if N%2 == 0:
        N += 1
    val = uniform_filter1d(val, size=N)
    return time, val

fit_func = lambda t, p: p[0] + p[1]*t + p[2]*t**3 + p[3]*t**5 + p[4]*t**7

def fit_range(time, amp, edge_ignore=0.1, frac=0.7, return_mask=False):
    mask_edge = np.logical_and(time > np.min(time) + edge_ignore,
                               time < np.max(time) - edge_ignore)

    thres = np.max(amp[mask_edge])*frac

    mask = np.logical_and(mask_edge, amp > thres)
    t_range = np.min(time[mask]), np.max(time[mask])
    if return_mask:
        return t_range, mask
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

def poly_fit(time, phi,time_range=[0,2]):
    mask = np.logical_and(time > np.min(time_range), time < np.max(time_range))

    chi2 = lambda p: np.sum((fit_func(time[mask], p) - phi[mask])**2)

    x0 = np.random.normal(size=5)
    x0[1] += 314
    res = minimize(chi2, x0)
    return res
