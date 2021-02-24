# -*- coding: utf-8 -*-
import numpy as np
from scipy import integrate
from ..units import *

class UnitVectorArray(object):
    """This class helps keeping track of different coordinate systems.
    The class has implemented these two systems by now
        1. x, y, z (cartesian coordinates)
        2. L (longitudinal amplitude), T (transversal amplitude), phase (in transversal plane)

    The relations between the coordinates are:
        L = y
        T = sqrt(x^2 + z^2)
        phase = arctan2(z, x), which means +x axis 0 deg, +z axis 90 deg, -x axis 180 deg and -y axis 270 deg
    """
    def __init__(self, x, y, z):
        self.set_x_y_z(x,y,z)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def L(self):
        return self._y

    @property
    def T(self):
        return np.sqrt(self._x**2 + self._z**2)

    @property
    def phase(self):
        return np.arctan2(self._z, self._x)

    def set_x_y_z(self, x, y, z):
        norm = np.sqrt(x**2+y**2+z**2)
        self._x = x/norm
        self._y = y/norm
        self._z = z/norm

    def set_L_T_phase(self, L, T, phase):
        self._x = T*np.cos(phase)
        self._y = L
        self._z = T*np.sin(phase)

class FID_simulation(object):
    def __init__(self, probe, b_field, N_cells, seed):
        self.B_field = b_field
        self.probe = probe
        self.rng = np.random.RandomState(seed)

        self.initialize_cells(N_cells)
        # by initializing we want to start in equalibrium
        self.equalibrium()

    def initialize_cells(self, N_cells):
        """Generates cells randomly distriubted in the probe sample.
        Calculates the external magnetic field at each cell point.
        Calculates the coil magnetic field at each cell point.
        Calculates the magnitude of the magnetization of each cell based on the
        external field.
        """
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
        # that is the amplitude not the orientation
        self.cells_magnetization = self.probe.magnetization(self.cells_B0)

    def frequency_spectrum(self):
        omega_mixed = (self.probe.material.gyromagnetic_ratio*self.cells_B0-2*np.pi*self.probe.mix_down)
        weights = np.sqrt(self.cells_B1_x**2+self.cells_B1_z**2)*self.cells_mu.T
        return omega_mixed, weights/np.mean(weights)

    def mean_frequency(self):
        f, w = self.frequency_spectrum()
        return np.average(f, weights=w)

    def std_frequency(self):
        f, w = self.frequency_spectrum()
        return np.sqrt(np.average((f-self.mean_frequency())**2, weights=w))

    def central_frequency(self):
        B0 = np.sqrt(np.sum(np.array(self.B_field(0, 0, 0))**2, axis=-1))
        return (self.probe.material.gyromagnetic_ratio*B0-2*np.pi*self.probe.mix_down)

    def equalibrium(self):
        # in equalibrium the magnetization points along the main field direction
        self.cells_mu = UnitVectorArray(self.cells_B0_x,
                                        self.cells_B0_y,
                                        self.cells_B0_z)

    # rename rf pulse ideal
    # that should not assume that we start in equalibrium,
    # so make it work for pulse duration of any time and any starting magnetization
    def apply_rf_field(self, time=None):
        ###
        ### angle of rotation axis: theta
        ### tan(theta) = B_RF / DeltaB
        ### DeltaB = - Omega/gamma
        ### Omega = omega_0 + omega_rf
        ### omega_0 = - gamma B0
        ### --> tan(theta) = gamma B_RF / (gamma B_0 - omega_RF)
        ###
        ### angle by which to rotate: beta
        ###
        if time is None:
            time = self.probe.estimate_rf_pulse()

        # aproximation
        L = np.cos(self.probe.material.gyromagnetic_ratio*self.cells_B1/2.*time)
        T = np.sin(self.probe.material.gyromagnetic_ratio*self.cells_B1/2.*time)
        phase = np.arctan(1./(self.probe.material.T2*self.probe.material.gyromagnetic_ratio*self.cells_B0))
        self.cells_mu.set_L_T_phase(L, T, phase)

    def spin_echo(self, time_pi=None, pretrigger=False, useBloch=True, pi_2_pulse_length=None, **kwargs):
        if time_pi is None:
            time_pi = self.probe.readout_length

        if pi_2_pulse_length is None:
            pi_2_pulse_length = self.probe.estimate_rf_pulse()

        # apply pi/2 pulse
        if useBloch:
            self.solve_bloch_eq_nummerical(time=pi_2_pulse_length,
                                           omega_rf=2*np.pi*self.probe.rf_pulse_frequency)
        else:
            self.apply_rf_field()

        # FID
        flux1, time1 = self.generate_FID(pretrigger=pretrigger, **kwargs)

        # apply pi pulse
        if useBloch:
            self.solve_bloch_eq_nummerical(time=2*pi_2_pulse_length,
                                           omega_rf=2*np.pi*self.probe.rf_pulse_frequency)
        else:
            self.cells_phase0 *= -1

        # spin echo
        t = np.arange(0, 2*time_pi, 1/self.probe.sampling_rate_offline)
        flux2, time2 = self.generate_FID(time=t, **kwargs)
        return np.concatenate([flux1, flux2]), np.concatenate([time1, time2+time_pi])

    # rename in free precession ideal
    def generate_FID(self, time=None, noise=None, max_memory=10000000, pretrigger=False):
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
            t = np.arange(0, self.probe.readout_length, 1/self.probe.sampling_rate_offline)
        else:
            t = np.atleast_1d(time)

        if pretrigger:
            N_pre =  int(self.probe.time_pretrigger*self.probe.sampling_rate_offline)
            t = t[:-N_pre]

        magnitude = np.sqrt((self.probe.material.gyromagnetic_ratio*self.cells_B0)**2 + 1/self.probe.material.T2**2)
        omega_mixed = (self.probe.material.gyromagnetic_ratio*self.cells_B0-2*np.pi*self.probe.mix_down)

        flux = []
        chunks = int(self.N_cells* len(t) / max_memory + 1)
        t0 = t[0]
        for this_t in np.array_split(t, chunks):
            this_t = this_t - t0
            argument = np.outer(omega_mixed,this_t) - self.cells_mu.phase[:, None]
            # this is equal to Bx * dmu_x_dt + By * dmu_y_dt + Bz * dmu_z_dt
            # already assumed that dmu_y_dt is 0, so we can leave out that term
            weight_x = self.cells_B1_x/np.mean(self.cells_B1)
            weight_z = self.cells_B1_z/np.mean(self.cells_B1)
            B_x_dmu_dt = self.cells_mu.T[:, None]*magnitude[:, None]*(weight_x[:, None]*np.cos(argument) + weight_z[:, None]*np.sin(argument))*(np.exp(-this_t/self.probe.material.T2)[:, None]).T
            #return self.coil.turns * mu0 * np.sum(B_x_dmu_dt/self.cells_B1[:, None]*self.cells_magnetization[:, None], axis=0) * np.pi * self.coil.radius**2
            flux.append(self.probe.coil.turns * mu0 * np.sum(B_x_dmu_dt*self.cells_magnetization[:, None], axis=0) * np.pi * self.probe.coil.radius**2)
            t0 += this_t[-1]
            self.cells_mu.set_L_T_phase(self.cells_mu.L,
                                        self.cells_mu.T * np.exp(-this_t[-1]/self.probe.material.T2),
                                        self.cells_mu.phase - omega_mixed*this_t[-1])
        flux = np.concatenate(flux)/self.N_cells

        if pretrigger:
            N_pre =  int(self.probe.time_pretrigger*self.probe.sampling_rate_offline)
            t = np.arange(0, self.probe.readout_length, 1/self.probe.sampling_rate_offline)
            flux = np.concatenate([np.zeros(N_pre), flux])
        if noise is not None:
            FID_noise = noise(t)
            flux += FID_noise
        return flux, t

    # rename bloch equation
    def solve_bloch_eq_nummerical(self, time=None, initial_condition=None, omega_rf=2*np.pi*61.79*MHz, with_relaxation=False, time_step=1.*ns, with_self_contribution=True, phase_rf=0):
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
            # latest state of mu
            initial_condition = [self.cells_mu.x, self.cells_mu.y, self.cells_mu.z]

        # pulse frequency
        def Bloch_equation(t, M):
            M = M.reshape((3, self.N_cells))
            Mx, My, Mz = M[0], M[1], M[2]

            Bx = self.cells_B0_x
            By = self.cells_B0_y
            Bz = self.cells_B0_z
            if with_self_contribution:
                Bx = Bx + mu0*self.cells_magnetization*Mx
                By = By + mu0*self.cells_magnetization*My
                Bz = Bz + mu0*self.cells_magnetization*Mz
            if omega_rf is not None:
                rf_osci = np.sin(omega_rf*t+phase_rf)
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

        rk_res = integrate.RK45(Bloch_equation,
                                t0=0,
                                y0=np.array(initial_condition).flatten(),
                                t_bound=time,
                                max_step=0.1*ns)  # about 10 points per oscillation
        history = []

        central_cell = np.argmin(self.cells_x**2 + self.cells_y**2 + self.cells_z**2)
        M = None
        while rk_res.status == "running":
            M = rk_res.y.reshape((3, self.N_cells))
            Mx, My, Mz = M[0], M[1], M[2]
            w = self.cells_B1/np.mean(self.cells_B1)
            history.append((rk_res.t, np.mean(w*Mx), np.mean(w*My), np.mean(w*Mz), Mx[central_cell], My[central_cell], Mz[central_cell]))
            rk_res.step()

        self.cells_mu.set_x_y_z(M[0], M[1], M[2])

        return np.array(history, dtype=[(k, np.float) for k in ["time", "Mx_mean", "My_mean", "Mz_mean", "Mx_center", "My_center", "Mz_center"]])
