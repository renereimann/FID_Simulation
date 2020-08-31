import numpy as np
from scipy import integrate
from ..units import *

class FID_simulation(object):
    def __init__(self, probe, b_field, N_cells, seed):
        self.B_field = b_field
        self.probe = probe
        self.rng = np.random.RandomState(seed)

        self.initialize_cells(N_cells)

    def initialize_cells(self, N_cells):
        # initialize cells
        self.N_cells = N_cells
        self.cells_x, self.cells_y, self.cells_z = self.probe.random_samples(self.rng, N_cells)

        # calculate external field at cell
        B0 = np.array([self.B_field(x, y, z) for x, y, z in zip(self.cells_x, self.cells_y, self.cells_z)])
        self.cells_B0_x = B0[:,0]
        self.cells_B0_y = B0[:,1]
        self.cells_B0_z = B0[:,2]
        self.cells_B0 = np.sqrt(np.sum(B0**2, axis=-1))

        # calculate coil field at cell
        B1 = np.array([self.probe.coil.B_field(x, y, z) for x, y, z in zip(self.cells_x, self.cells_y, self.cells_z)])
        self.cells_B1_x = B1[:,0]
        self.cells_B1_y = B1[:,1]
        self.cells_B1_z = B1[:,2]
        self.cells_B1 = np.sqrt(np.sum(B1**2, axis=-1))

        # calculate magnetization of cells
        # dipoles are aligned with the external field at the beginning
        self.cells_magnetization = self.probe.magnetization(self.cells_B0)
        #self.cells_dipole_moment_mag = self.cells_magnetization * self.probe.V_cell/N_cells

    def apply_rf_field(self, time=None):
        if time is None:
            time = self.probe.estimate_rf_pulse()

        # aproximation
        self.cells_mu_x = np.sin(self.probe.material.gyromagnetic_ratio*self.cells_B1/2.*time)
        self.cells_mu_y = np.cos(self.probe.material.gyromagnetic_ratio*self.cells_B1/2.*time)
        self.cells_mu_z = np.sin(self.probe.material.gyromagnetic_ratio*self.cells_B1/2.*time)
        self.cells_mu_T = np.sqrt(self.cells_mu_x**2 + self.cells_mu_z**2)
        self.cells_phase0 = np.arctan(1./(self.probe.material.T2*self.probe.material.gyromagnetic_ratio*self.cells_B0))

    def spin_echo(self, mix_down=0*MHz, time_pi=None):
        if time_pi is None:
            time_pi = self.probe.readout_length
        # apply pi/2 pulse
        self.apply_rf_field()

        # FID
        t = np.arange(0, time_pi, 1/self.probe.sampling_rate_offline)
        flux1, time1 = self.generate_FID(time=t, mix_down=mix_down)

        # apply pi pulse
        self.cells_phase0 *= -1

        # spin echo
        t = np.arange(0, 2*time_pi, 1/self.probe.sampling_rate_offline)
        flux2, time2 = self.generate_FID(time=t, mix_down=mix_down)
        return np.concatenate([flux1, flux2]), np.concatenate([time1, time2+time_pi])

    def generate_FID(self, time=None, mix_down=0*MHz, useAverage=True, noise=None, max_memory=10000000):
        # pickup_flux is depricated and generate_FID should be used instead.
        # Return typ is different. pickup_flux only returned flux and expected a
        # time series, while generate_FID can default to a time series and Returns
        # both flux and time series
        (r"""
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

        #######################
        # * EMF ->  see Ran's "MUON G-2 NMR FREQUENCY EXTRACTION" Sec. 5.1 Eq. 36:
        #           potential energy U = - current * Phi_muC
        #          Phi_muC = vec(mu) dot_product vec(B_coil) / current
        #
        #          --> units
        #            mu = A m²
        #            B = T
        #            current = A
        #            --> U = J or eV
        #            That corresponds for electrons with charge e a voltage in V.
        """)

        t = None
        if time is None:
            t = np.arange(0, self.probe.readout_length-self.probe.time_pretrigger,
                          1/self.probe.sampling_rate_offline)
            #t = np.linspace(0*ms, 10*ms, 10000) # 1 MSPS
        else:
            t = np.atleast_1d(time)

        if not hasattr(self, "cells_mu_T"):
            self.apply_rf_field()

        magnitude = np.sqrt((self.probe.material.gyromagnetic_ratio*self.cells_B0)**2 + 1/self.probe.material.T2**2)
        omega_mixed = (self.probe.material.gyromagnetic_ratio*self.cells_B0-2*np.pi*mix_down)

        flux = []
        chunks = int(self.N_cells* len(t) / max_memory + 1)
        for this_t in np.array_split(t, chunks):
            this_t = this_t-this_t[0]
            argument = np.outer(omega_mixed,this_t) - self.cells_phase0[:, None]
            # this is equal to Bx * dmu_x_dt + By * dmu_y_dt + Bz * dmu_z_dt
            # already assumed that dmu_y_dt is 0, so we can leave out that term
            B_x_dmu_dt = self.cells_mu_T[:, None]*magnitude[:, None]*(self.cells_B1_x[:, None]*np.cos(argument) + self.cells_B1_z[:, None]*np.sin(argument))*(np.exp(-this_t/self.probe.material.T2)[:, None]).T
            #return self.coil.turns * µ0 * np.sum(B_x_dmu_dt/self.cells_B1[:, None]*self.cells_magnetization[:, None], axis=0) * np.pi * self.coil.radius**2
            if useAverage:
                flux.append(self.probe.coil.turns * µ0 * np.sum(B_x_dmu_dt*self.cells_magnetization[:, None], axis=0) * np.pi * self.probe.coil.radius**2 /np.mean(self.cells_B1[:, None]))
            else:
                flux.append(self.probe.coil.turns * µ0 * np.sum(B_x_dmu_dt*self.cells_magnetization[:, None]/self.cells_B1[:, None], axis=0) * np.pi * self.probe.coil.radius**2)
            # Alternative
            # flux.append(µ0 * np.mean(B_x_dmu_dt*self.cells_magnetization[:, None], axis=0) )/(self.cells_B1[:, None])
            self.cells_phase0 -= omega_mixed*this_t[-1]
            self.cells_mu_T *= np.exp(-this_t[-1]/self.probe.material.T2)
        flux = np.concatenate(flux)/self.N_cells

        if time is None:
            t_pre = np.arange(0, self.probe.time_pretrigger, 1/self.probe.sampling_rate_offline)
            f_pre = np.zeros_like(t_pre)
            t = np.concatenate([t_pre, t+self.probe.time_pretrigger])
            flux = np.concatenate([f_pre, flux])
        if noise is not None:
            FID_noise = noise(t)
            flux += FID_noise
        return flux, t

    def solve_bloch_eq_nummerical(self, time=None, initial_condition=None, omega_rf=2*np.pi*61.79*MHz, with_relaxation=False, time_step=1.*ns, with_self_contribution=True):
        (r"""Solves the Bloch Equation numerically for a RF pulse with length `time`
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
        """)

        if time is None:
            # if no time is given we estimate the pi/2 pulse duration and use that
            time = self.estimate_rf_pulse()

        if initial_condition is None:
            # if no initial condition for the magnetization is given, we use the
            # equilibrium magnetization, which is aligned with the direction of
            # the external field.
            initial_condition = [self.cells_B0_x/self.cells_B0,
                                 self.cells_B0_y/self.cells_B0,
                                 self.cells_B0_z/self.cells_B0]

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
            dMx = self.probe.material.gyromagnetic_ratio*(My*Bz-Mz*By)
            dMy = self.probe.material.gyromagnetic_ratio*(Mz*Bx-Mx*Bz)
            dMz = self.probe.material.gyromagnetic_ratio*(Mx*By-My*Bx)
            if with_relaxation:
                # note we approximate here that the external field is in y direction
                # in the ideal case we would calculate the B0_field direct and the ortogonal plane
                # note that we use relative magnetization , so the -1 is -M0
                dMx -= Mx/self.probe.material.T2
                dMy -= (My-1)/self.probe.material.T1
                dMz -= Mz/self.probe.material.T2
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
