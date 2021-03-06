# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import hilbert
from ..units import uV, ms

class HilbertTransform(object):
    def __init__(self, times, flux):
        if len(times) != len(flux):
            raise AttributeError("times and flux must have the same dimension")
        self.N = len(flux)
        self.time = times / ms
        self.flux = flux / uV
        self.h = hilbert(self.flux)

    def real(self):
        return np.real(self.h)

    def imag(self):
        return np.imag(self.h)

    def PhaseFunction(self):
        phi = np.arctan(self.imag()/self.real())
        jump = np.pi*np.logical_and(phi[:-1] > 0, phi[1:] < 0)
        phi += np.concatenate([[0], np.cumsum(jump)])
        return self.time*ms, phi

    def EnvelopeFunction(self):
        return self.time*ms, np.sqrt(self.imag()**2 + self.real()**2)*uV

    def plot_phase_function(self, fig=None):
        if fig is None:
            fig, ax = plt.subplots()

        t, phi = self.PhaseFunction()
        ax.scatter(t, np.degrees(phi))
