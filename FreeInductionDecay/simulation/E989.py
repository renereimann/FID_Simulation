from ..units import *
from .probe import NMRProbe
from .coil import Coil
from .material import PetroleumJelly, PP_Water
from .magnet import RingMagnet

class FixedProbe(NMRProbe):
    def __init__(self):
        self.readout_length = 4*ms
        self.sampling_rate_online = 10*MHz
        self.sampling_rate_offline = 1*MHz
        self.time_pretrigger = 409*us
        fix_probe_coil = Coil(turns=30,
                              length=15.0*mm,
                              diameter=4.6*mm,
                              current=0.7*A)
        super().__init__(length = 30.0*mm,
                         diameter = 1.5*mm,
                         material = PetroleumJelly,
                         temp = (273.15 + 26.85) * K,
                         coil = fix_probe_coil)


class PlungingProbe(NMRProbe):
    def __init__(self):

        self.readout_length = 500*ms
        self.sampling_rate_online = 10*MHz
        self.sampling_rate_offline = self.sampling_rate_online
        self.time_pretrigger = 0 # ???
        # L = 0.5 uH
        # C_p = 1-12 pF
        # C_s = 1-12 pF in series with L*C_p
        plunging_probe_coil = Coil(turns=5.5,
                              length=10.0*mm,
                              diameter=15.065*mm+(0.97*mm/2),
                              current=0.7*A)

        super().__init__(length = 228.6*mm,
                         diameter = 4.2065*mm,
                         material = PP_Water,
                         temp = (273.15 + 26.85) * K,
                         coil = plunging_probe_coil)


class StorageRingMagnet(RingMagnet):
    def __init__(self, B0=1.45*T):
        super().__init__(B0)
