from FreeInductionDecay.units import *
from FreeInductionDecay.simulation.E989 import StorageRingMagnet, FixedProbe
from FreeInductionDecay.simulation.FID_sim import FID_simulation
from FreeInductionDecay.simulation.noise import WhiteNoise
​
import numpy as np
import matplotlib.pyplot as plt
​
​# create an external magnetic field and set a linear gradient with strength 10 ppm​
b_field = StorageRingMagnet( ) # 1,45 T
B0 = b_field.An[2]
b_field.An[8] = 10*ppm/cm*B0

# create an NMR probe
probe = FixedProbe()

# setup the simulation with 1000 cells and seed 1
sim = FID_simulation(probe, b_field, N_cells=1000, seed=1)
​
# setup noise. The noise scale is 2% of the maximal FID signal amplitude
# ToDo: Make a better example here
noise = WhiteNoise(scale=12*uV*0.02)

# Simulate the solution of the Bloch equation while applying an RF field for 14µs.
data = np.array(sim.solve_bloch_eq_nummerical(time=14*us, omega_rf=2*np.pi*61.79*MHz))
​
​# We plot the magnetization on a 3D sphere and the projections on the x, y, z plane.
plot_RF_pulse_3D(data)
plt.savefig("./Test_Bloch_Equation.pdf", bbox_inches="tight", metadata={"Autor": "FreeInductionDecay Simulation Package",
                                                                       "Title": "Magnetization while an RF field. Simulated using Bloch Equations."})​

fig = plt.figure()
step = 1
plt.scatter(data[::step,4], data[::step,6], c=data[::step,0]/us,
                marker="o", cmap="jet", vmin=data[0,0]/us, vmax=data[-1,0]/us)
plt.colorbar().set_label("time / $\mu$s")
plt.savefig("./Test_Bloch_Equation_x_z_projection.pdf", bbox_inches="tight", metadata={"Autor": "FreeInductionDecay Simulation Package",
                                                                       "Title": "Magnetization while an RF field. Simulated using Bloch Equations. Projection on xz plane."})​
