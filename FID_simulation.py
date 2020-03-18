# Based on DocDB # 16856
# https://gm2-docdb.fnal.gov/cgi-bin/private/RetrieveFile?docid=16856&filename=fid_simulation.pdf&version=4

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Constants go here
mu0 = 1.256637062e-6 # N/A^2

# Geometry
probe_total_length = 100.00 #mm
probe_total_diameter = 8.00 #mm

# Fixed Probe
coil_diameter = 2.3 #mm
coil_length = 15.0 #mm
coil_turns = 30

cell_type = "cylinder"
cell_length = 30.0 #mm
cell_diameter = 1.5 #mm

# material
probe_material = "Petroleum Jelly"
T2 = 40*1e-3 # sec # how mature are these values
#gamma =  # gyromagnetic ratio of the spin sample particles.

# magnetic field
B0 = 1.45 # T, static field strength
# B1 = # RF field strength
f_NMR = 61.79 # MHz
I = 5 #A

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

coil_path_parameter = np.linspace(0, 2*np.pi*coil_turns, 1000)

lx = lambda phi: coil_diameter/2. * np.sin(phi)
ly = lambda phi: coil_diameter/2. * np.cos(phi)
lz = lambda phi: coil_length * phi / (2*np.pi*coil_turns) - coil_length/2

dlx = lambda phi: coil_diameter/2. * np.cos(phi)
dly = lambda phi: -coil_diameter/2. * np.sin(phi)
dlz = lambda phi: coil_length / (2*np.pi*coil_turns)

dist = lambda phi, x, y, z: np.sqrt((lx(phi)-x)**2+(ly(phi)-y)**2+(lz(phi)-z)**2)
integrand_x = lambda phi, x, y, z: ( dly(phi) * (z-lz(phi)) - dlz(phi) * (y-ly(phi)) ) / dist(phi, x, y, z)**3
integrand_y = lambda phi, x, y, z: ( dlz(phi) * (x-lx(phi)) - dlx(phi) * (z-lz(phi)) ) / dist(phi, x, y, z)**3
integrand_z = lambda phi, x, y, z: ( dlx(phi) * (y-ly(phi)) - dly(phi) * (x-lx(phi)) ) / dist(phi, x, y, z)**3

B1_x = lambda x, y, z : mu0/(4*np.pi) * I * integrate.quad(lambda phi: integrand_x(phi, x, y, z), 0, 2*np.pi*coil_turns)[0]
B1_y = lambda x, y, z : mu0/(4*np.pi) * I * integrate.quad(lambda phi: integrand_y(phi, x, y, z), 0, 2*np.pi*coil_turns)[0]
B1_z = lambda x, y, z : mu0/(4*np.pi) * I * integrate.quad(lambda phi: integrand_z(phi, x, y, z), 0, 2*np.pi*coil_turns)[0]

if True:
    # make a plot for comparison
    cross_check = np.genfromtxt("./RF_coil_field_cross_check.txt", delimiter=", ")
    zs = np.linspace(-15, 15, 1000)
    plt.figure()
    plt.plot(cross_check[:,0], cross_check[:,1], label="Cross-Check from DocDB 16856, Slide 5\n$\O=2.3\,\mathrm{mm}$, $L=15\,\mathrm{mm}$, turns=30", color="orange")
    B1_z0 = B1_z(0, 0, 0)
    plt.plot(zs, [B1_z(0, 0, z)/B1_z0 for z in zs], label="My calculation\n$\O=2.3\,\mathrm{mm}$, $L=15\,\mathrm{mm}$, turns=30", color="k")
    coil_diameter = 2*2.3 # mm
    B1_z0 = B1_z(0, 0, 0)
    plt.plot(zs, [B1_z(0, 0, z)/B1_z0 for z in zs], label="My calculation\n$\O=4.6\,\mathrm{mm}$, $L=15\,\mathrm{mm}$, turns=30", color="k", ls=":")
    plt.xlabel("z / mm")
    plt.ylabel("$B_z(0,0,z)\, /\, B_z(0,0,0)$")
    plt.legend(loc="lower right")
    plt.title("Magnetic field of the coil (static)")
    plt.tight_layout()
    plt.savefig("./plots/coil_field_distribution.pdf", bbox_inches="tight")

########################################################################################

# apply RF field
# B1 field strength of RF field
# for time of t_pi2 = pi/(2 gamma B1)
'''
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
'''
