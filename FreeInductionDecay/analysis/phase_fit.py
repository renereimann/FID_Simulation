import numpy as np
from scipy.optimize import minimize
from scipy.ndimage.filters import uniform_filter1d
from ..units import *
from .hilbert_transform import HilbertTransform
import matplotlib.pyplot as plt

class PhaseFitFID(object):
    def __init__(self, probe=None, edge_ignore=0.1*ms, frac=0.7, smoothing=True, window_size=1/(50*kHz), tol=1e-5):
        self.t0 = probe.time_pretrigger
        self.pretrigger = probe.time_pretrigger
        self.readout_length = probe.readout_length
        self.edge_ignore = edge_ignore
        self.frac = frac
        self.window_size = window_size
        self.smoothing = smoothing
        self.tol = tol
        self.fit_window_fact = 1

    def get_fit_range(self):
        t_min = np.min(self.time)
        t_max = np.max(self.time)
        if self.pretrigger is not None:
            t_min = np.max([t_min, self.pretrigger])
        if self.readout_length is not None:
            t_max = np.min([t_max, self.readout_length])
        mask_edge = np.logical_and(self.time > t_min + self.edge_ignore,
                                   self.time < t_max - self.edge_ignore)

        thres = np.max(self.env[mask_edge])*self.frac
        mask = np.logical_and(mask_edge, self.env > thres)

        t_min = np.min(self.time[mask])
        t_max = np.min(self.time[np.logical_and(self.time > t_min, np.logical_not(mask))])
        return np.array([t_min, t_max])

    def fit_func(self, t, p):
        return p[0] + p[1]*t + p[2]*t**3 + p[3]*t**5 + p[4]*t**7
        #return p[0] + p[1]*t + p[2]*t**2 + p[3]*t**3 + p[4]*t**4+ p[5]*t**5 + p[6]*t**6 + p[7]*t**7

    def apply_smoothing(self):
        N = int(self.window_size/np.diff(self.time)[0])
        if N%2 == 0:
            N += 1
        return uniform_filter1d(self.phase_raw, size=N)

    def get_noise(self):
        return np.std(self.flux[self.time < self.pretrigger])

    def chi2_fit(self):
        mask = np.logical_and(self.time > np.min(self.t_range), self.time < np.max(self.t_range))
        self.width = (self.t_range[1]-self.t_range[0])/self.fit_window_fact
        chi2 = lambda p: np.sum((self.fit_func((self.time[mask]-self.t0)/self.width, p) - self.phase[mask])**2*(self.env[mask]/self.noise)**2)
        x0 = np.random.normal(scale=0.1, size=5)
        x0[1] += 314
        res = minimize(chi2, x0, tol=self.tol, method="L-BFGS-B")
        return res

    def fit(self, time, flux):
        self.time = time
        self.flux = flux
        hilbert = HilbertTransform(self.time, self.flux)
        _, self.env =  hilbert.EnvelopeFunction()
        _, self.phase_raw =  hilbert.PhaseFunction()
        self.noise = self.get_noise()
        if self.smoothing:
            self.phase = self.apply_smoothing()
        else:
            self.phase = self.phase_raw[:]
        self.t_range = self.get_fit_range()

        self.res = self.chi2_fit()
        self.n_point_in_fit = np.sum(np.logical_and(self.time > np.min(self.t_range), self.time < np.max(self.t_range)))
        self.frequency = self.res.x[1]/self.width
        self.phi0 = self.res.x[0]

        return self.frequency

    def plot(self):
        plt.plot(self.time/ms, self.phase_raw - self.phi0 - self.frequency*(self.time-self.t0), color="b", label="raw FID")
        if self.smoothing:
            plt.plot(self.time/ms, self.phase - self.phi0 - self.frequency*(self.time-self.t0), color="red", label="smoothed FID")
            plt.errorbar(self.time/ms, self.phase - self.phi0 - self.frequency*(self.time-self.t0), yerr=self.noise/self.env, color="red", label="smoothed FID")
        #plt.plot(time[mask]/ms, phi_fit[mask] - phi0 - frequency*(time[mask]-t0), color="k", ls="--", label="fit")
        phase_fit = self.fit_func((self.time-self.t0)/self.width, self.res.x)
        plt.plot(self.time/ms, phase_fit - self.phi0 - self.frequency*(self.time-self.t0), color="k", ls="--", label="fit")
        plt.grid()
        plt.axvspan(*(self.t_range/ms), color="gray", alpha=0.2)
        plt.ylim(-0.1, 0.2)
        plt.xlabel("time / ms")
        plt.xlim( (np.mean(self.t_range)-0.55*(self.t_range[1]-self.t_range[0]))/ms, (np.mean(self.t_range)+0.55*(self.t_range[1]-self.t_range[0]))/ms)
        plt.ylabel(r"$ \Phi(t) - \hat{\Phi_0} - \hat{\frac{\mathrm{d}\Phi}{\mathrm{d}t}}\cdot t$")
        plt.legend()
        plt.axvline(self.t0/ms, ls="--", color="k")
        plt.text(self.t0/ms, 0.20, "trigger", rotation=90, va="top", ha="left", fontsize=12, fontweight='bold')
        plt.xlim(xmin=self.t0/ms*0.95)

class PhaseFitEcho(PhaseFitFID):
    def __init__(self, frac=0.7, probe=None, smoothing=True, window_size=1/(50*kHz), tol=1e-5):
        self.t0 = 2*probe.readout_length-probe.time_pretrigger
        self.pretrigger = probe.time_pretrigger
        self.readout_length = probe.readout_length
        self.frac = frac
        self.window_size = window_size
        self.smoothing = True
        self.tol = tol
        self.fit_window_fact = 1

    def fit_func(self, t, p):
        # the phase function needs also even components
        return p[0] + p[1]*t + p[2]*t**2 + p[3]*t**3 + p[4]*t**4

    def get_fit_range(self):
        # closest index to t0
        idx = np.argmin(np.abs(self.time - self.t0))
        # threshold relative to t0
        thres = self.frac * self.env[idx]

        t_start = np.max(self.time[np.logical_and(self.time < self.t0, self.env < thres)])
        t_stop = np.min(self.time[np.logical_and(self.time > self.t0, self.env < thres)])
        return np.array([t_start, t_stop])

    def plot(self):
        plt.plot(self.time/ms, self.phase_raw - self.phi0 - self.frequency*(self.time-self.t0), color="b", label="raw FID")
        if self.smoothing:
            plt.plot(self.time/ms, self.phase - self.phi0 - self.frequency*(self.time-self.t0), color="red", label="smoothed FID")
        #plt.plot(time[mask]/ms, phi_fit[mask] - phi0 - frequency*(time[mask]-t0), color="k", ls="--", label="fit")
        phase_fit = self.fit_func((self.time-self.t0)/self.width, self.res.x)
        plt.plot(self.time/ms, phase_fit - self.phi0 - self.frequency*(self.time-self.t0), color="k", ls="--", label="fit")
        plt.grid()
        plt.axvspan(*(self.t_range/ms), color="gray", alpha=0.2)
        plt.ylim(-0.1, 0.2)
        plt.xlabel("time / ms")
        plt.xlim( (np.mean(self.t_range)-0.55*(self.t_range[1]-self.t_range[0]))/ms, (np.mean(self.t_range)+0.55*(self.t_range[1]-self.t_range[0]))/ms)
        plt.ylabel(r"$ \Phi(t) - \hat{\Phi_0} - \hat{\frac{\mathrm{d}\Phi}{\mathrm{d}t}}\cdot t$")
        plt.legend()
        plt.axvline(self.t0/ms, ls="--", color="k")
        plt.text(self.t0/ms, 0.20, "Echo", rotation=90, va="top", ha="left", fontsize=12, fontweight='bold')
