# -*- coding: utf-8 -*-

import numpy as np
from scipy import integrate
from ..units import *

class Coil(object):
    r"""A coil parametrized by number of turns, length, diameter and current.

    You can calculate the magnetic field cause by the coil at any point in space.
    """
    def __init__(self, turns, length, diameter, current):
        r""" Generates a coil objsect.

        Parameters:
        * turns: int
        * length: float
        * diameter: float
        * current: float
        """
        self.turns = turns
        self.length = length
        self.radius = diameter/2.
        self.current = current

    def B_field(self, x, y, z):
        r"""The magnetic field of the coil
        Assume Biot-Savart law
        vec(B)(vec(r)) = µ0 / 4π ∮ I dvec(L) × vec(r)' / |vec(r)'|³

        Approximations:
            - static, Biot-Savart law only holds for static current,
              in case of time-dependence use Jefimenko's equations.
              Jefimenko's equation:
              vec(B)(vec(r)) = µ0 / 4π ∫ ( J(r',tᵣ)/|r-r'|³ + 1/(|r-r'|² c)  ∂J(r',tᵣ)/∂t) × (r-r') d³r'
              with t_r = t-|r-r'|/c
              --> |r-r'|/c of order 0.1 ns
            - infinite small cables, if cables are extendet use the dV form
              vec(B)(vec(r))  = µ0 / 4π ∭_V  (vec(J) dV) × vec(r)' / |vec(r)'|³
            - closed loop
            - constant current, can factor out the I from integral
        """

        phi = np.linspace(0, 2*np.pi*self.turns, 10000)
        sPhi = np.sin(phi)
        cPhi = np.cos(phi)
        lx = self.radius*sPhi
        ly = self.radius*cPhi
        lz = self.length/2 * (phi/(np.pi*self.turns)-1)
        dlx = ly
        dly = -lx
        dlz = self.length/(2*np.pi*self.turns)

        dist = np.sqrt((lx-x)**2+(ly-y)**2+(lz-z)**2)

        integrand_x = ( dly * (z-lz) - dlz * (y-ly) ) / dist**3
        integrand_y = ( dlz * (x-lx) - dlx * (z-lz) ) / dist**3
        integrand_z = ( dlx * (y-ly) - dly * (x-lx) ) / dist**3

        B_x = mu0/(4*np.pi) * self.current * integrate.simps(integrand_x, x=phi)
        B_y = mu0/(4*np.pi) * self.current * integrate.simps(integrand_y, x=phi)
        B_z = mu0/(4*np.pi) * self.current * integrate.simps(integrand_z, x=phi)

        return [B_x, B_y, B_z]

    def Bz(self, z):
        """ This is an analytical solution for the B_z component along the x=y=0
        axis. We used the formula from "Experimentalphysik 2" Demtröder Section
        3.2.6 d) (Page 95/96, 5th edition)
        """
        n = self.turns / self.length
        I = self.current
        L = self.length
        R = self.coil.radius
        B_z = lambda z: mu0*n*I/2*((z+L/2)/np.sqrt(R**2+(z+L/2)**2)-(z-L/2)/np.sqrt(R**2+(z-L/2)**2))
        return B_z(z)
