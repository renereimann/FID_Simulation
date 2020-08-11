from ..units import *
from .probe import Coil, NMRProbe
from .material import PetroleumJelly, PP_Water
from .magnet import RingMagnet

class FixedProbe(NMRProbe):
    def __init__(self, B_field, N_cells=1000, seed=12345):
        current = 0.7*A # ???
        fix_probe_coil = Coil(turns=30,
                              length=15.0*mm,
                              diameter=4.6*mm,
                              current=current)
        super().__init__(length = 30.0*mm,
                         diameter = 1.5*mm,
                         material = PetroleumJelly,
                         temp = (273.15 + 26.85) * K,
                         B_field = B_field,
                         coil = fix_probe_coil,
                         N_cells = N_cells,
                         seed = seed)


class PlungingProbe(NMRProbe):
    def __init__(self, B_field, N_cells=1000, seed=12345):

        # L = 0.5 uH
        # C_p = 1-12 pF
        # C_s = 1-12 pF in series with L*C_p
        current = 0.7*A # ???

        plunging_probe_coil = Coil(turns=5.5,
                              length=10.0*mm,
                              diameter=15.065*mm+(0.97*mm/2),
                              current=current)

        super().__init__(length = 228.6*mm,
                         diameter = 4.2065*mm,
                         material = PP_Water,
                         temp = (273.15 + 26.85) * K,
                         B_field = B_field,
                         coil = plunging_probe_coil,
                         N_cells = N_cells,
                         seed = seed)


class StorageRingMagnet(RingMagnet):
    def __init__(self, B0=1.45*T):
        super().__init__(B0)
