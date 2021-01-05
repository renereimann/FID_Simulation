# -*- coding: utf-8 -*-
import numpy as np
from ..units import *

class Material(object):
    """ An Material instance holds material related properties """
    def __init__(self, name, formula=None, density=None, molar_mass=None, T1=None, T2=None, gyromagnetic_ratio=None):
        """Generate Material instance

        Parameters:
        * name: str, Name of the Material
        * formula: str, chemical formula of the Material
        * density: float, density of the material, e.g. in units of g/cm^3
        * molar_mass: float, molare Masse of the Material, e.g. in units of g/mol
        * T1: float, longitudinal relaxation time of the Material, e.g. in s
        * T2: float, transversal relaxation time of the Material, e.g. in s
        * gyromagnetic_ratio: float, gyromagnetic ration of protons shifted by material effects
        """
        # gyromagnetic ratio, value for free proton: 2.6752218744e8*Hz/T
        # magnetic moment,  value for free proton: 1.41060679736e-26*J/T
        self.name = name
        self.formula = formula
        self.density = density
        self.molar_mass = molar_mass
        self.T1 = T1
        self.T2 = T2
        self.gyromagnetic_ratio = gyromagnetic_ratio
        self.magnetic_moment = self.gyromagnetic_ratio*hbar/2

    def __str__(self):
        """String representation of the Material for pretty printing."""
        info = []
        if self.formula is not None:
            info.append(self.formula)
        if self.density is not None:
            info.append("%f g/cm^3"%(self.density/(g/cm**3)))
        if self.molar_mass is not None:
            info.append("%f g/mol"%(self.molar_mass/(g/mol)))
        if self.T1 is not None:
            info.append("%f ms"%(self.T1/ms))
        if self.T2 is not None:
            info.append("%f ms"%(self.T2/ms))
        if self.gyromagnetic_ratio is not None:
            info.append("%f Hz/T"%(self.gyromagnetic_ratio/(Hz/T)))
        return self.name + "(" + ", ".join(info) + ")"

    @property
    def number_density(self):
        # NA * density / molar_mass
        # the package numericalunits already converts "mol" using the Avrogardo
        # constant thus we do not need the extra factor if using numericalunits
        return self.density / self.molar_mass


PetroleumJelly = Material(name = "Petroleum Jelly",
                           formula = "C40H46N4O10",
                           density = 0.848*g/cm**3,
                           molar_mass = 742.8*g/mol,
                           T1 = 1*s,
                           T2 = 40*ms,
                           gyromagnetic_ratio=(2*np.pi)*61.79*MHz/(1.45*T),
                           )

sigma_H2O = lambda T: 25691e-9 - 10.36e-9*(T-(25*K+T0))/K
mag_susceptibility_H2O = lambda T: -9049e-9*(1 + 1.39e-4*(T-(20*K+T0))/K - 1.27e-7 *(T/K-(20+T0/K))**2 + 8.09e-10 *(T/K-(20+T0/K))**3 )
delta_b_H2O = lambda T: (0.49991537 - (1/3)) * mag_susceptibility_H2O(T)
omega_p_meas = lambda T: 2.6752218744e8*Hz/T*(1 - sigma_H2O(T) - delta_b_H2O(T) + 5.5*ppb )
PP_Water = Material(name = "Ultra-Pure ASTM Type 1 Water",
                           formula = "H2O",
                           density = 997*kg/m**3,
                           molar_mass = 18.01528*g/mol,
                           T1 = 3*s,
                           T2 = 3*s,
                           gyromagnetic_ratio=omega_p_meas(300*K),
                           )
