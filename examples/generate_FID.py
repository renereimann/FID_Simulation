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
​
# run a spin echo sequence.
# The readout window has a pretrigger window and we add noise to the simulation
# You get back the flux vs time
flux_echo, time_echo = sim.spin_echo(pretrigger=True, noise=noise)
​
​# Plotting the FID signal
fig, ax = plt.subplots()
ax.set_xlabel("time in ms")
ax.set_ylabel("Amplitude in a.u.")
ax.set_xlim(0, 12)
ax.set_ylim(-20, 20)
ax.plot(time_echo/ms, flux_echo/uV, color="blue")
ax.text(0.5, 2, "$\pi$/2 pulse", rotation=90, horizontalalignment='left', verticalalignment='bottom')
ax.text(4.1, 2, "$\pi$ pulse", rotation=90, horizontalalignment='left', verticalalignment='bottom')
ax.text(7.9, 2, "Spin Echo", rotation=90, horizontalalignment='left', verticalalignment='bottom')
ax.axvline(sim.probe.time_pretrigger/ms, ls="--", color="k")
ax.axvline(sim.probe.readout_length/ms, ls="--", color="k")
ax.axvline((2*sim.probe.readout_length-sim.probe.time_pretrigger)/ms, ls="--", color="k")
fig.savefig("./Test_FID_generation.pdf", bbox_inches="tight", metadata={"Autor": "FreeInductionDecay Simulation Package",
                                                                        "Title": "Test FreeInductionDecay Signal Simulation",
                                                                        "Subject": "This FID signal is generated with the FreeInductionDecay simulation package. It assumes an external field of 1.45T with a 10ppm linear gradient. The probe is made out of a coil and filled with petroleum jelly. We run a spin echo pulse sequence on it. Here we plot the pickup voltage vs time over a typical readout window."})
