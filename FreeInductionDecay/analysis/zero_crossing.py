# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import UnivariateSpline

# David Flay has implemented that in a recursive / iterative procedure (including)
# baseliine correction (up to a precision of 1e-16 s)
# See Field Paper Sec. II B 2

# There is a second definition of ZeroCrossing that is used for baseline extraction
# This is described in Rans internal note and followed below

class ZeroCrossing(object):
    def __init__(self, times, flux):
        if len(times) != len(flux):
            raise AttributeError("times and flux must have the same dimension")
        self.N = len(flux)
        self.time = times
        self.flux = flux

    def linear_intersect(self, x1, y1, x2, y2, y=0):
        return (y - y1)/self.derivative(x1, y1, x2, y2) + x1

    def derivative(self, x1, y1, x2, y2):
        return (y2-y1)/(x2-x1)

    def PhaseFunction(self):
        zero_crossings = []
        derivatives = []
        # sign can be -1, 0 or 1
        sign = np.sign(self.flux)
        for i in range(len(self.flux)-1):
            # if sign is 0, we have hit the zero exactly --> use this time as zero crossing
            # slope from previous and next datapoint
            if sign[i] == 0:
                k = self.derivative(self.time[i-1], self.flux[i-1],
                                    self.time[i+1], self.flux[i+1])
                zero_crossings.append(self.time[i])
                derivatives.append(k)
            elif sign[i] == -sign[i+1]:
                t0 = self.linear_intersect(self.time[i], self.flux[i],
                                      self.time[i+1], self.flux[i+1])
                k = self.derivative(self.time[i], self.flux[i],
                                    self.time[i+1], self.flux[i+1])
                zero_crossings.append(t0)
                derivatives.append(k)
        self.zero_crossings = np.array(zero_crossings)
        self.derivatives = np.array(derivatives)
        self.phi = (np.arange(len(self.zero_crossings))+0.5)*np.pi
        return self.zero_crossings, self.phi

    def baseline_spline(self):
        self.PhaseFunction()
        t = self.zero_crossings
        k = self.derivatives
        baseline = []
        for i in range(1, len(t)-1):
            yb = (2*t[i] - t[i-1] - t[i+1]) / (1/k[i-1] + 1/k[i+1]-2/k[i])
            baseline.append(yb)
        self.baseline = np.array(baseline)
        base_spline = UnivariateSpline(t[1:-1][np.isfinite(baseline)], self.baseline[np.isfinite(baseline)], k=1, s=0, ext=1)
        return base_spline

    def get_asymmetry(self, return_pos_neg=False):
        Ap = []
        An = []
        is_pos = lambda f: f > 0
        sign = is_pos(self.flux[0])
        t_max = self.time[0]
        f_max = np.abs(self.flux[0])
        for t, f in zip(self.time[1:], self.flux[1:]):
            if sign == is_pos(f):
                if np.abs(f) > f_max:
                    t_max = t
                    f_max = np.abs(f)
            else:
                if sign == True:
                    Ap.append((t_max, f_max))
                else:
                    An.append((t_max, f_max))
                sign = is_pos(f)
                t_max = t
                f_max = np.abs(f)
        Ap_spl = UnivariateSpline([t for t, A in Ap], [A for t, A in Ap], k=1, s=0)
        An_spl = UnivariateSpline([t for t, A in An], [A for t, A in An], k=1, s=0)
        As = lambda t: Ap_spl(t)/An_spl(t) - 1
        if return_pos_neg:
            return Ap, An
        return As
