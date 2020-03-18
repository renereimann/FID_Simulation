# Based on DocDB # 16856
# https://gm2-docdb.fnal.gov/cgi-bin/private/RetrieveFile?docid=16856&filename=fid_simulation.pdf&version=4

import numpy as np
import matplotlib.pyplot as plt
# Constants go here

# Geometry
probe_total_length = 100.00 #mm
probe_total_diameter = 8.00 #mm
coil_diameter = 2.3 #mm
coil_length = 15.0 #mm
coil_turns = 30
cell_type = "cylinder"
cell_length = 30.0 #mm
cell_diameter = 1.5 #mm

# material
probe_material = "Petroleum Jelly"
T2 = 40*1e-3 # sec # how mature are these values
gamma =  # gyromagnetic ratio of the spin sample particles.

# magnetic field
B0 = 1.45 # T, static field strength
B1 = # RF field strength
f_NMR = 61.79 # MHz

# simulation
seed = 12345
N_particles = 1000

########################################################################################
# that about motion / diffusion within material
# what about other components of the probe
# what about spin-spin interactions


# setup random number
rng = np.random.RandomState(seed)

# setup particles and spins
r = np.sqrt(np.random.uniform(0,cell_diameter**2/4, size=N_particles))
phi = np.random.uniform(0, 2*np.pi, size=N_particles)
z = np.random.uniform(-cell_length/2., cell_length/2., size=N_particles)
x = r*np.sin(phi)
y = r*np.cos(phi)

spin_cosTheta = np.random.uniform(-1, 1, size=N_particles)
spin_phi = np.random.uniform(0, 2*np.pi, size=N_particles)

########################################################################################

# the field of the coil
# Assume Biot-Savart law
# vec(B)(vec(r)) = mu0 / 4pi Int_C I dvec(L) x vec(r)' / |vec(r)'|^3
# Approximations:
#      - static, Biot-Savart law only holds for static current,
#        in case of time-dependence use Jefimenko's equations.
#        Jefimenko's equation:
#        vec(B)(vec(r)) = mu0 / 4pi Int ( J(r',t_r)/|r-r'|^3 + 1/(|r-r'|^2 c)*partial J(r', t_r)/partial t)  x (r-r') d^3r'
#        with t_r = t-|r-r'|/c
#        --> |r-r'|/c of order 0.1 ns
#      - infinite small cables, if cables are extendet use the dV form of Biot-Savart
#        vec(B)(vec(r)) = mu0 / 4pi IntIntInt_V (vec(J)dV) x vec(r)' / |vec(r)'|^3
#      - closed loop
#      - constant current, can factor out the I from integral

########################################################################################

# apply RF field
# B1 field strength of RF field
# for time of t_pi2 = pi/(2 gamma B1)

t_90 = np.pi/(2*gamma*B1)

# spin
mu_x(vec(r),t) = np.sin(gamma*B1(vec(r))*t)*np.cos(gamma*B0(vec(r))*t)
mu_y(vec(r),t) = np.cos(gamma*B1(vec(r))*t)
mu_z(vec(r),t) = np.sin(gamma*B1(vec(r))*t)*np.sin(gamma*B0(vec(r))*t)

########################################################################################

# Let the spin precess
# T2 transversal relaxation

mu_T = np.sqrt(mu_x**2 + mu_y**2)
mu_x(t) = - mu_T * np.cos(gamma * B0(vec(r))*t)*np.exp(-t/T2)
mu_y(t) = mu_y(t_0)
mu_z(t) = mu_T * np.sin(gamma * B0(vec(r))*t)*np.exp(-t/T2)

# Add longitudinal relaxation (T1)

########################################################################################

# Flux through pickup coil
# sensitivity of coil
phi(t) = np.sum(coil_turns*vec(B2(vec(r_mu)))*vec(mu(t)))/I1

########################################################################################

# integrate dM/dt with RK4
