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

    def Bz(self, z):
        """ This is an analytical solution for the B_z component along the x=y=0
        axis. We used the formula from "Experimentalphysik 2" Demtröder Section
        3.2.6 d) (Page 95/96, 5th edition)
        """
        n = self.turns / self.length
        I = self.current
        L = self.length
        R = self.coil.radius
        B_z = lambda z: µ0*n*I/2*((z+L/2)/np.sqrt(R**2+(z+L/2)**2)-(z-L/2)/np.sqrt(R**2+(z-L/2)**2))
        return B_z(z)


class NMRProbe(object):
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

    def apply_rf_field(self, time=None):
        if time is None:
            time = self.t_90_estimate()

        if not hasattr(self, "cells_B1"):
            self.initialize_coil_field()

        # aproximation
        self.cells_mu_x = np.sin(self.material.gyromagnetic_ratio*self.cells_B1/2.*time)
        self.cells_mu_y = np.cos(self.material.gyromagnetic_ratio*self.cells_B1/2.*time)
        self.cells_mu_z = np.sin(self.material.gyromagnetic_ratio*self.cells_B1/2.*time)
        self.cells_mu_T = np.sqrt(self.cells_mu_x**2 + self.cells_mu_z**2)

    def solve_bloch_eq_nummerical(self,
                                  time=None,
                                  initial_condition=None,
                                  omega_rf=2*np.pi*61.79*MHz,
                                  with_relaxation=False,
                                  time_step=1.*ns,
                                  with_self_contribution=True):
        """Solves the Bloch Equation numerically for a RF pulse with length `time`
        and frequency `omega_rf`.

        Parameters:
            * time: Length of RF pulse (e.g. a pi/2 pulse time or pi pulse time)
                    If time is None, the pi/2 is estimated.
                    Default: None
            * initial_condition: Inital mu_x, mu_y, mu_z of cells.
                    Given as arrays of shap [3,N_cells]
                    If None the inital condition is assumed to be in equilibrium
                    and calculated from external B_field
                    Default: None
            * omega_rf: RF pulse frequency
                    If omega_rf is None, no RF pulse is applied and a Free
                    Induction Decay is happening
                    Default: 2 pi 61.79 MHz
            * with_relaxation: If true the relaxation terms are considered in the
                    Bloch equations. If false the relaxation terms are neglected.
                    Default: False
            * time_step: Float, maximal time step used in nummerical solution of
                    the Differential equation. Note that this value should be
                    sufficient smaller than the oscillation time scale.
                    Default: 1 ns
            * with_self_contribution: Boolean, if True we consider the additional
                    B-field from the magnetization of the cell.
                    Default: True
        Returns:
            * history: array of shape (7, N_time_steps)
                       times, mean_Mx, mean_My, mean_Mz, Mx(0,0,0), My(0,0,0), Mz(0,0,0)


        Note: all magnetizations are treated as relative parameters wrt to the
              equalibrium magnetization, i.e. all values are without units and
              restricted to -1 and 1.
        """

        if time is None:
            # if no time is given we estimate the pi/2 pulse duration and use that
            time = self.t_90_estimate()

        if initial_condition is None:
            # if no initial condition for the magnetization is given, we use the
            # equilibrium magnetization, which is aligned with the direction of
            # the external field.
            initial_condition = [self.cells_B0_x/self.cells_B0,
                                 self.cells_B0_y/self.cells_B0,
                                 self.cells_B0_z/self.cells_B0]

        if not hasattr(self, "cells_B1"):
            # if cells B1 is not yet calculated, calculate B1 components
            self.initialize_coil_field()

        # pulse frequency
        def Bloch_equation(t, M):
            M = M.reshape((3, self.N_cells))
            Mx, My, Mz = M[0], M[1], M[2]

            Bx = self.cells_B0_x
            By = self.cells_B0_y
            Bz = self.cells_B0_z
            if with_self_contribution:
                Bx = Bx + µ0*self.cells_magnetization*Mx
                By = By + µ0*self.cells_magnetization*My
                Bz = Bz + µ0*self.cells_magnetization*Mz
            if omega_rf is not None:
                rf_osci = np.sin(omega_rf*t)
                Bx = Bx + rf_osci * self.cells_B1_x
                By = By + rf_osci * self.cells_B1_y
                Bz = Bz + rf_osci * self.cells_B1_z
            dMx = self.material.gyromagnetic_ratio*(My*Bz-Mz*By)
            dMy = self.material.gyromagnetic_ratio*(Mz*Bx-Mx*Bz)
            dMz = self.material.gyromagnetic_ratio*(Mx*By-My*Bx)
            if with_relaxation:
                # note we approximate here that the external field is in y direction
                # in the ideal case we would calculate the B0_field direct and the ortogonal plane
                # note that we use relative magnetization , so the -1 is -M0
                dMx -= Mx/self.material.T2
                dMy -= (My-1)/self.material.T1
                dMz -= Mz/self.material.T2
            return np.array([dMx, dMy, dMz]).flatten()

        #solution = integrate.odeint(Bloch_equation,
        #                            y0=np.array(initial_condition).flatten(),
        #                            t=np.linspace(0., time, int(time/ns)))

        rk_res = integrate.RK45(Bloch_equation,
                                t0=0,
                                y0=np.array(initial_condition).flatten(),
                                t_bound=time,
                                max_step=0.1*ns)  # about 10 points per oscillation
        history = []

        idx = np.argmin(self.cells_x**2 + self.cells_y**2 + self.cells_z**2)
        M = None
        while rk_res.status == "running":
            M = rk_res.y.reshape((3, self.N_cells))
            Mx, My, Mz = M[0], M[1], M[2]                                       # 1
            #wx = self.cells_B1_x/np.sum(np.sort(self.cells_B1_x))
            #wy = self.cells_B1_y/np.sum(np.sort(self.cells_B1_y))
            #wz = self.cells_B1_z/np.sum(np.sort(self.cells_B1_z))
            #history.append([rk_res.t, np.sum(np.sort(Mx*wx)), np.sum(np.sort(My*wy)), np.sum(np.sort(Mz*wz)), Mx[idx], My[idx], Mz[idx]])
            history.append([rk_res.t, np.mean(Mx), np.mean(My), np.mean(Mz), Mx[idx], My[idx], Mz[idx]])
            rk_res.step()

        self.cells_mu_x = M[0]
        self.cells_mu_y = M[1]
        self.cells_mu_z = M[2]
        self.cells_mu_T = np.sqrt(self.cells_mu_x**2 + self.cells_mu_z**2)

        return history

    def t_90_estimate(self):
        brf = self.coil.B_field(0*mm,0*mm,0*mm)
        # B1 field strength is half of RF field
        b1 = np.sqrt(brf[0]**2+brf[1]**2+brf[2]**2)/2.
        t_90 = (np.pi/2)/(self.material.gyromagnetic_ratio*b1)
        return t_90

    def generate_FID(self, t=None, mix_down=0*MHz, useAverage=True, noise=None):
        if t is None:
            t = np.linspace(0*ms, 10*ms, 10000) # 1 MSPS

        flux = self.pickup_flux(t, mix_down=mix_down, useAverage=useAverage)
        if noise is not None:
            FID_noise = noise(t)
            flux += FID_noise
        return flux, t


    def pickup_flux(self, t, mix_down=0*MHz, useAverage=True):
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

        if not hasattr(self, "cells_mu_x"):
            self.apply_rf_field()

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
            if useAverage:
                results.append(self.coil.turns * µ0 * np.sum(B_x_dmu_dt*self.cells_magnetization[:, None], axis=0) * np.pi * self.coil.radius**2 /np.mean(self.cells_B1[:, None]))
            else:
                results.append(self.coil.turns * µ0 * np.sum(B_x_dmu_dt*self.cells_magnetization[:, None]/self.cells_B1[:, None], axis=0) * np.pi * self.coil.radius**2)
            # Alternative
            # results.append(µ0 * np.mean(B_x_dmu_dt*self.cells_magnetization[:, None], axis=0) )/(self.cells_B1[:, None])
        results = np.concatenate(results)/N_cells
        return results
