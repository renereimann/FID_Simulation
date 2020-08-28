# This script simulates the FID signal of a pNMR probe.
#
# Author: René Reimann (2020)
#
# The ideas are based on DocDB #16856 and DocDB #11289
# https://gm2-docdb.fnal.gov/cgi-bin/private/ShowDocument?docid=16856
# https://gm2-docdb.fnal.gov/cgi-bin/private/ShowDocument?docid=11289

################################################################################
# Import first

import numpy as np
from scipy import integrate
from ..units import *

class NMRProbe(object):
    def __init__(self, length, diameter, material, temp, coil):
        self.length = length
        self.radius = diameter / 2.
        self.V_cell = self.length * np.pi * self.radius**2

        self.material = material
        self.temp = temp
        self.coil = coil

    def magnetization(self, B_field):
        """Calculates the probes magnetization for a given B field value.
        B_field can be an array, in which case magnetization for each entry are calculated"""
        expon = self.material.magnetic_moment / (kB*self.temp) * B_field
        nuclear_polarization = (np.exp(expon) - np.exp(-expon))/(np.exp(expon) + np.exp(-expon))
        magnetizations = self.material.magnetic_moment * self.material.number_density * nuclear_polarization
        return magnetizations

    def random_samples(self, rng, size):
        r = np.sqrt(rng.uniform(0,self.radius**2, size=size))
        phi = rng.uniform(0, 2*np.pi, size=size)
        x = r*np.sin(phi)
        y = r*np.cos(phi)
        z = rng.uniform(-self.length/2., self.length/2., size=size)
        return x, y, z

    def estimate_rf_pulse(self, alpha=np.pi/2):
        brf = self.coil.B_field(0*mm,0*mm,0*mm)
        # B1 field strength is half of RF field
        b1 = np.sqrt(brf[0]**2+brf[1]**2+brf[2]**2)/2.
        t_alpha = alpha/(self.material.gyromagnetic_ratio*b1)
        return t_alpha
