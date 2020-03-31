# This script simulates the FID signal of a pNMR probe.
#
# Author: René Reimann (2020)
#
# The ideas are based on DocDB #16856 and DocDB #11289
# https://gm2-docdb.fnal.gov/cgi-bin/private/ShowDocument?docid=16856
# https://gm2-docdb.fnal.gov/cgi-bin/private/ShowDocument?docid=11289

################################################################################
# Import first

import time
import numpy as np
from scipy import integrate
from scipy import fftpack
from numericalunits import µ0, NA, kB, hbar, mm, cm, m, s, ms, us, Hz, kHz, MHz
from numericalunits import T, K, J, g, mol, A, ohm, W, N, kg, mV, V, eV, uV
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as mcolors

################################################################################

class PermanentMagnet(object):
    def __init__(self, B0):
        self.An = self.P = { # dipoles
                   1: 0*T,
                   2: B0,
                   3: 0*T,
                   # quadrupoles
                   4: 0*T/mm,
                   5: 0*T/mm,
                   6: 0*T/mm,
                   7: 0*T/mm,
                   8: 0*T/cm,
                   # sextupoles
                   9: 0*T/mm**2,
                  10: 0*T/mm**2,
                  11: 0*T/mm**2,
                  12: 0*T/mm**2,
                  13: 0*T/mm**2,
                  14: 0*T/mm**2,
                  15: 0*T/mm**2,
                  # octupole
                  16: 0*T/mm**3,
                  17: 0*T/mm**3,
                  18: 0*T/mm**3,
                  19: 0*T/mm**3,
                  20: 0*T/mm**3,
                  21: 0*T/mm**3,
                  22: 0*T/mm**3,
                  23: 0*T/mm**3,
                  24: 0*T/mm**3,
                 }

        self.P = { # dipoles
                   1: {"x": lambda x, y, z: 1, "y": lambda x, y, z: 0, "z": lambda x, y, z: 0},
                   2: {"x": lambda x, y, z: 0, "y": lambda x, y, z: 1, "z": lambda x, y, z: 0},
                   3: {"x": lambda x, y, z: 1, "y": lambda x, y, z: 0, "z": lambda x, y, z: 1},
                   # quadrupoles
                   4: {"x": lambda x, y, z: x, "y": lambda x, y, z: -y, "z": lambda x, y, z: 0},
                   5: {"x": lambda x, y, z: z, "y": lambda x, y, z: 0, "z": lambda x, y, z: x},
                   6: {"x": lambda x, y, z: 0, "y": lambda x, y, z: -y, "z": lambda x, y, z: z},
                   7: {"x": lambda x, y, z: y, "y": lambda x, y, z: x, "z": lambda x, y, z: 0},
                   8: {"x": lambda x, y, z: 0, "y": lambda x, y, z: z, "z": lambda x, y, z: y},
                   # sextupoles
                   9: {"x": lambda x, y, z: x**2-y**2, "y": lambda x, y, z: -2*x*y, "z": lambda x, y, z: 0},
                  10: {"x": lambda x, y, z: 2*x*z, "y": lambda x, y, z: -2*y*z, "z": lambda x, y, z: x**2-y**2},
                  11: {"x": lambda x, y, z: z**2-y**2, "y": lambda x, y, z: -2*x*y, "z": lambda x, y, z: 2*x*y},
                  12: {"x": lambda x, y, z: 0, "y": lambda x, y, z: -2*y*z, "z": lambda x, y, z: z**2-y**2},
                  13: {"x": lambda x, y, z: 2*x*y, "y": lambda x, y, z: x**2-y**2, "z": lambda x, y, z: 0},
                  14: {"x": lambda x, y, z: y*z, "y": lambda x, y, z: x*z, "z": lambda x, y, z: x*y},
                  15: {"x": lambda x, y, z: 0, "y": lambda x, y, z: z**2-y**2, "z": lambda x, y, z: 2*y*z},
                  # octupole
                  16: {"x": lambda x, y, z: x**3 - 3*x*y**2, "y": lambda x, y, z: y**3-3*x**2*y, "z": lambda x, y, z: 0},
                  17: {"x": lambda x, y, z: 3*x**2*z-3*z*y**2, "y": lambda x, y, z: -6*x*y*z, "z": lambda x, y, z: x**3 - 3*x*y**2},
                  18: {"x": lambda x, y, z: 3*x*z**2-3*x*y**2, "y": lambda x, y, z: -3*x**2*y-3*z**2*y+2*y**3, "z": lambda x, y, z: 3*x**2*z-3*z*y**2},
                  19: {"x": lambda x, y, z: z**3-3*z*y**2, "y": lambda x, y, z: -6*x*y*z, "z": lambda x, y, z: 3*x*z**2 - 3*x*y**2},
                  20: {"x": lambda x, y, z: 0, "y": lambda x, y, z: y**3-3*z**2*y, "z": lambda x, y, z: z**3-3*z*y**2},
                  21: {"x": lambda x, y, z: 3*x**2*y-y**3, "y": lambda x, y, z: x**3-3*x*y**2, "z": lambda x, y, z: 0},
                  22: {"x": lambda x, y, z: 6*x*y*z, "y": lambda x, y, z: 3*x**2*z-3*z*y**2, "z": lambda x, y, z: 3*x**2*y-y**3},
                  23: {"x": lambda x, y, z: 3*z**2*y-y**3, "y": lambda x, y, z: 3*x*z**2-3*x*y**2, "z": lambda x, y, z: 6*x*y*z},
                  24: {"x": lambda x, y, z: 0, "y": lambda x, y, z: z**3-3*z*y**2, "z": lambda x, y, z: 3*z**2*y-y**3},
                 }

    def B_field(self, x=0, y=0, z=0):
        Bx = 0
        By = 0
        Bz = 0
        for i in self.P.keys():
            Bx += self.An[i]*self.P[i]["x"](x, y, z)
            By += self.An[i]*self.P[i]["y"](x, y, z)
            Bz += self.An[i]*self.P[i]["z"](x, y, z)
        return [Bx, By, Bz]

    def __call__(self, x=0, y=0, z=0):
        return self.B_field(x, y, z)


class Material(object):
    def __init__(self, name, formula=None, density=None, molar_mass=None, T1=None, T2=None, gyromagnetic_ratio=None):
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

        B_x = µ0/(4*np.pi) * self.current * integrate.simps(integrand_x, x=phi)
        B_y = µ0/(4*np.pi) * self.current * integrate.simps(integrand_y, x=phi)
        B_z = µ0/(4*np.pi) * self.current * integrate.simps(integrand_z, x=phi)

        return [B_x, B_y, B_z]


class Probe(object):
    def __init__(self, length, diameter, material, temp, B_field, coil, N_cells, seed):
        self.length = length
        self.radius = diameter / 2.
        self.V_cell = self.length * np.pi * self.radius**2

        self.material = material

        self.temp = temp
        self.B_field = B_field
        self.coil = coil

        self.rng = np.random.RandomState(seed)
        self.N_cells = N_cells

        self.initialize_cells(self.N_cells)

    def initialize_cells(self, N_cells):
        # place cells
        r = np.sqrt(self.rng.uniform(0,self.radius**2, size=N_cells))
        phi = self.rng.uniform(0, 2*np.pi, size=N_cells)
        self.cells_x = r*np.sin(phi)
        self.cells_y = r*np.cos(phi)
        self.cells_z = self.rng.uniform(-self.length/2., self.length/2., size=N_cells)

        # calculate quantities of cells
        B0 = np.array([self.B_field(x, y, z) for x, y, z in zip(self.cells_x, self.cells_y, self.cells_z)])
        self.cells_B0_x = B0[:,0]
        self.cells_B0_y = B0[:,1]
        self.cells_B0_z = B0[:,2]
        self.cells_B0 = np.sqrt(np.sum(B0**2, axis=-1))

        expon = self.material.magnetic_moment * self.cells_B0 / (kB*self.temp)
        self.cells_nuclear_polarization = (np.exp(expon) - np.exp(-expon))/(np.exp(expon) + np.exp(-expon))
        self.cells_magnetization = self.material.magnetic_moment * self.material.number_density * self.cells_nuclear_polarization
        # dipoles are aligned with the external field at the beginning
        self.cells_dipole_moment_mag = self.cells_magnetization * self.V_cell/N_cells

    def initialize_coil_field(self):
        # clculate B-field from coil for each cell
        B1 = np.array([self.coil.B_field(x, y, z) for x, y, z in zip(self.cells_x, self.cells_y, self.cells_z)])
        self.cells_B1_x = B1[:,0]
        self.cells_B1_y = B1[:,1]
        self.cells_B1_z = B1[:,2]
        self.cells_B1 = np.sqrt(np.sum(B1**2, axis=-1))

    def apply_rf_field(self, time):
        if not hasattr(self, "cells_B1"):
            self.initialize_coil_field()

        # aproximation
        self.cells_mu_x = np.sin(self.material.gyromagnetic_ratio*self.cells_B1/2.*time)
        self.cells_mu_y = np.cos(self.material.gyromagnetic_ratio*self.cells_B1/2.*time)
        self.cells_mu_z = np.sin(self.material.gyromagnetic_ratio*self.cells_B1/2.*time)
        self.cells_mu_T = np.sqrt(self.cells_mu_x**2 + self.cells_mu_z**2)

    def t_90(self):
        brf = self.coil.B_field(0*mm,0*mm,0*mm)
        # B1 field strength is half of RF field
        b1 = np.sqrt(brf[0]**2+brf[1]**2+brf[2]**2)/2.
        t_90 = (np.pi/2)/(self.material.gyromagnetic_ratio*b1)
        return t_90

    def pickup_flux(self, t, mix_down=0*MHz):
        # Φ(t) = Σ N B₂(r) * μ(t) / I
        # a mix down_frequency can be propergated through and will effect the
        # individual cells

        # flux in pickup coil depends on d/dt(B × μ)
        # --> y component static, no induction, does not contribute

        # d/dt ( μₜ sin(γₚ |B0| t) exp(-t/T2) )
        #       = μₜ [ d/dt( sin(γₚ |B0| t) ) exp(-t/T2) + sin(γₚ |B0| t) d/dt( exp(-t/T2) )]
        #       = μₜ [ γₚ |B0| cos(γₚ |B0| t) exp(-t/T2) + sin(γₚ |B0| t) (-1/T2) exp(-t/T2) ]
        #       = μₜ [ γₚ |B0| cos(γₚ |B0| t) -1/T2 * sin(γₚ |B0| t) ] exp(-t/T2)
        # make use of Addition theorem a cos(α) + b sin(α) = √(a² + b²) cos(α - arctan(-b/a))
        #       = μₜ √(γₚ² |B0|² + 1/T2²) cos(γₚ |B0| t - arctan(1/(T2γₚ |B0|)) exp(-t/T2)

        # a mix down_frequency can be propergated through and will effect the
        # individual cells, all operations before are linear
        # Note the mix down will only effect the

        # straight forward implementation
        # very inefficient
        # mu_x = lambda cell : cell.mu_T*np.sin((γₚ*cell.B0-mix_down)*t)*np.exp(-t/self.material.T2)
        # dmu_x_dt = lambda cell: cell.mu_T*np.sqrt((γₚ*cell.B0)**2 + 1/self.material.T2**2)*np.cos((γₚ*cell.B0-mix_down)*t - np.arctan(1/(self.material.T2*(γₚ*cell.B0)))*np.exp(-t/self.material.T2)
        # mu_y = lambda cell : cell.mu_z
        # dmu_y_dt = lambda cell: 0
        # mu_z = lambda cell : cell.mu_T*np.cos((γₚ*cell.B0-mix_down)*t)*np.exp(-t/self.material.T2)
        # dmu_z_dt = lambda cell: cell.mu_T*np.sqrt((γₚ*cell.B0)**2 + 1/self.material.T2**2)*np.sin((γₚ*cell.B0-mix_down)*t - np.arctan(1/(self.material.T2*(γₚ*cell.B0)))*np.exp(-t/self.material.T2)
        # return np.sum( [cell.B1.x * dmu_x_dt(cell) + cell.B1.y * dmu_y_dt(cell) + cell.B1.z * dmu_z_dt(cell) for cell in self.cells] )

        # From Faradays law we have
        # EMF = - d/dt (N * B * A)
        #     with vec(B)(t) = mu0 * vec(M)(t) = mu0 * M0 * vec(mu)(t)
        #     with vec(A) = pi*r^2 * vec(B_coil)/<B_coil>
        # EMF = N * pi * r^2 * mu0 * sum( vec(B_coil)/<B_coil> * M0 * d/dt( vec(mu)(t) ) )

        t = np.atleast_1d(t)

        magnitude = self.cells_mu_T*np.sqrt((self.material.gyromagnetic_ratio*self.cells_B0)**2 + 1/self.material.T2**2)
        phase = np.arctan(1./(self.material.T2*self.material.gyromagnetic_ratio*self.cells_B0))
        omega_mixed = (self.material.gyromagnetic_ratio*self.cells_B0-2*np.pi*mix_down)

        max_memory = 10000000
        N_cells = len(self.cells_B0)
        results = []
        idx_end = 0
        while idx_end != -1:
            idx_start = idx_end
            this_t = None
            if N_cells* len(t[idx_start:]) > max_memory:
                idx_end = idx_start + max_memory//N_cells
                this_t = t[idx_start:idx_end]
            else:
                idx_end = -1
                this_t = t[idx_start:]

            argument = np.outer(omega_mixed,this_t) - phase[:, None]
            # this is equal to Bx * dmu_x_dt + By * dmu_y_dt + Bz * dmu_z_dt
            # already assumed that dmu_y_dt is 0, so we can leave out that term
            B_x_dmu_dt = magnitude[:, None]*(self.cells_B1_x[:, None]*np.cos(argument) + self.cells_B1_z[:, None]*np.sin(argument))*(np.exp(-this_t/self.material.T2)[:, None]).T
            #return self.coil.turns * µ0 * np.sum(B_x_dmu_dt/self.cells_B1[:, None]*self.cells_magnetization[:, None], axis=0) * np.pi * self.coil.radius**2
            results.append(self.coil.turns * µ0 * np.sum(B_x_dmu_dt*self.cells_magnetization[:, None], axis=0) * np.pi * self.coil.radius**2 /np.mean(self.cells_B1[:, None]))
        results = np.concatenate(results)/N_cells
        return results


class Noise(object):
    r"""Class to generate noise and drift for time series. We plan to support
    different kind of noise and drift.
    """

    def __init__(self, white_noise=None, freq_power=None, scale_freq=None,
                 drift_lin=None,
                 drift_exp=None, drift_exp_time=None, rng=None):
        r"""Creates a Noise object that can be called to generate noise for time
        series.

        Parameters:
            - freq_power = Power of the f^alpha Power density spectrum. Default 1
            - drift_lin = strength of linear drift
            - drift_exp = strength of exponential dirft
            - rng = RandomState object used to generate random numbers.
        """
        self.freq_power = freq_power
        self.scale_freq = scale_freq
        self.white_noise = white_noise
        self.drift_lin = drift_lin
        self.drift_exp = drift_exp
        self.drift_exp_time = drift_exp_time
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.RandomState()

    def get_freq_noise(self, times, rng):
        N = len(times)
        rand_noise = rng.normal(loc=0.0, scale=self.scale_freq, size=N)
        freq = np.fft.fftfreq(N, d=times[1]-times[0])
        fft  = fftpack.fft(rand_noise)
        fft[freq!=0] *= np.power(np.abs(freq[freq!=0]), 0.5*self.freq_power)
        fft[freq==0] = 0
        noise = fftpack.ifft(fft)
        return np.real(noise)

    def get_white_noise(self, times, rng):
        return rng.normal(loc=0.0, scale=self.white_noise)

    def get_linear_drift(self, times):
        return times*self.drift_lin

    def get_exp_drift(self, times):
        return  self.drift_exp*np.exp(-times/self.drift_exp_time)

    def __call__(self, times, rng=None):
        if rng is None: rng = self.rng
        noise = np.zeros_like(times)
        if self.freq_power is not None and self.scale_freq is not None:
            noise += self.get_freq_noise(times, rng=rng)
        if self.white_noise is not None:
            noise += self.get_white_noise(times, rng=rng)
        if self.drift_lin is not None:
            noise += self.get_linear_drift(times)
        if self.drift_exp is not None and self.drift_exp_time is not None:
            noise += self.get_exp_drift(times)
        return noise


################################################################################
# setup

impedance = 50 * ohm
guete = 1.1
pulse_power = 10*W
current = guete * np.sqrt(2*pulse_power/impedance)
print("I =", current/A, "A")

# values from wolframalpha.com
petroleum_jelly = Material(name = "Petroleum Jelly",
                           formula = "C40H46N4O10",
                           density = 0.848*g/cm**3,
                           molar_mass = 742.8*g/mol,
                           T1 = 1*s,
                           T2 = 40*ms,
                           gyromagnetic_ratio=(2*np.pi)*61.79*MHz/(1.45*T),
                           )
print(petroleum_jelly)

# external field
B0 = PermanentMagnet( 1.45*T )
B0.An[8] = 5e-6*B0.An[2]/cm

# NMR Coil
nmr_coil = Coil(turns=30,
               length=15.0*mm,
               diameter=4.6*mm,
               current=current)

# NMR Probe
nmr_probe = Probe(length = 30.0*mm,
                  diameter = 1.5*mm,
                  material = petroleum_jelly,
                  temp = (273.15 + 26.85) * K,
                  B_field = B0,
                  coil = nmr_coil,
                  N_cells = 10000,
                  seed = 12345)

noise = Noise(freq_power=-1, scale_freq=3*uV)
################################################################################
# calculation

# apply RF field
t_90 = nmr_probe.t_90()
print("t_90", t_90/us, "mus")

t_start = time.time()
nmr_probe.apply_rf_field(t_90)
print("Needed", (time.time()-t_start), "sec to calculate RF field.")

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

exit()

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
yf = fftpack.fft(flux*windowFunction[""](np.linspace(0,1,len(flux))))
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
    hilbert = fftpack.ifft(complex(0,-1)*np.sign(xf)*yf)
    print(hilbert)
    print(hilbert/yf)
    phi = np.arctan(hilbert/yf)
    plt.figure()
    plt.plot(times/ms, phi)

"""
# that about motion / diffusion within material
# what about other components of the probe
# what about spin-spin interactions


# would have to solve Bloch Equations
#                             ( B_RF sin(omega*t) )
# dvec(M)/dt = gamma vec(M) x (         0         )
#                             (         B_z       )

Bz = B0
Brf = rnm_coil.B_field_mag(0*mm,0*mm,0*mm)
omega_NMR = 61.79*MHz    # circuit of the probe tuned for this value

def Bloch_equation_with_RF_field(t, M):
    dM_dt = γₚ*np.cross(M, [Brf*np.sin(omega_NMR*t), Brf*np.cos(omega_NMR*t), Bz])
    return dM_dt

rk_res = integrate.RK45(Bloch_equation_with_RF_field,
                        t0=0,
                        y0=[0.3,0.3,0.3],
                        t_bound=t_90,
                        max_step=t_90/100000)
history = []
while rk_res.status == "running":
    history.append([rk_res.t, rk_res.y])
    rk_res.step()

####################################################################################

def Bloch_equation_with_RF_field(M, t, γₚ, Bz, Brf, omega, T1=np.inf, T2=np.inf):
    # M is a vector of length 3 and is: M = [Mx, My, My].
    # Return dM_dt, that is a vector of length 3 as well.
    Mx, My, Mz = M
    M0 = 1
    #relaxation = np.array([-Mx/T2, -My/T2, -(Mz-M0)/T1])

    dM_dt = γₚ*np.cross(M, [Brf*np.sin(omega*t), Brf*np.cos(omega*t), Bz]) #+ relaxation
    return dM_dt

solution = integrate.odeint(Bloch_equation_with_RF_field,
                            y0=[0.3, 0.3, 0.3],
                            t=np.linspace(0., t_90, 100000),
                            args=(γₚ, B0, nmr_coil.B_field_mag(0*mm,0*mm,0*mm), omega_NMR ))


plt.figure()
plt.plot([h[1][0] for h in history], label="$M_x$ (RK45)")
plt.plot(solution[:,0], color="r", ls="--", label="$M_x$ (odeint)")
plt.plot(solution[:,1], color="b", label="$M_y$ (odeint)")
plt.plot(solution[:,2], color="g", label="$M_z$ (odeint)")
plt.xlabel("time / steps")
plt.ylabel("$M_i$")
plt.legend()
plt.show()

########################################################################################

# Add longitudinal relaxation (T1)

# integrate dM/dt with RK4
# dMx/dt = -γₚ(My*Bz-Mz*By) - Mx/T2
# dMy/dt = -γₚ(Mz*Bx-Mx*Bz) - My/T2
# dMz/dt = -γₚ(Mx*By-My*Bx) - (Mz-M0)/T1

# limit T2 --> infty: Mx/y(t) = exp(-t/T1) sin(omega t-phi0)

########################################################################################
"""
