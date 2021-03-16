# -*- coding: utf-8 -*-

from ..units import *
from .probe import NMRProbe
from .coil import Coil
from .material import PetroleumJelly, PP_Water
from .magnet import RingMagnet

class FixedProbe(NMRProbe):
    def __init__(self, use_biot_savart=False):
        self.readout_length = 4.096*ms
        self.sampling_rate_online = 10*MHz
        self.sampling_rate_offline = 1*MHz
        self.time_pretrigger = 409*us
        self.mix_down = 61.74*MHz
        self.rf_pulse_frequency = 61.79*MHz
        fix_probe_coil = Coil(turns=32, # 0-27 equally distributed, 28-29 at one end, 30-31 at the other
                              length=15.0*mm,
                              diameter=4.6*mm,
                              current=0.7*A,
                              use_biot_savart=use_biot_savart)
        super(FixedProbe, self).__init__(length = 33.5*mm,
                         diameter = 2.5*mm,
                         material = PetroleumJelly,
                         temp = (273.15 + 26.85) * K,
                         coil = fix_probe_coil)

class TrolleyProbe(NMRProbe):
    def __init__(self):
        self.readout_length = 16*ms
        self.sampling_rate_online = 61.74*MHz/62
        self.sampling_rate_offline = self.sampling_rate_online
        self.time_pretrigger = 0.3*ms
        self.mix_down = 61.74*MHz
        self.rf_pulse_frequency = 61.79*MHz
        trolley_coil = Coil(turns=18, # two layers a 9 turns
                              length=7.0*mm,
                              diameter=4.6*mm,
                              current=0.7*A)

        super(TrolleyProbe, self).__init__(length = 33.5*mm,
                         diameter = 2.5*mm,
                         material = PetroleumJelly,
                         temp = (273.15 + 26.85) * K,
                         coil = trolley_coil)

class PlungingProbe(NMRProbe):
    def __init__(self):

        self.readout_length = 500*ms
        self.sampling_rate_online = 10*MHz
        self.sampling_rate_offline = self.sampling_rate_online
        self.time_pretrigger = 0 # ???
        self.mix_down = 61.74*MHz
        self.rf_pulse_frequency = 61.79*MHz
        # L = 0.5 uH
        # C_p = 1-12 pF
        # C_s = 1-12 pF in series with L*C_p
        plunging_probe_coil = Coil(turns=5.5,
                              length=10.0*mm,
                              diameter=15.065*mm+(0.97*mm/2),
                              current=0.7*A)

        super(PlungingProbe, self).__init__(length = 228.6*mm,
                         diameter = 4.2065*mm,
                         material = PP_Water,
                         temp = (273.15 + 26.85) * K,
                         coil = plunging_probe_coil)


class StorageRingMagnet(RingMagnet):
    def __init__(self, B0=1.45*T):
        super(StorageRingMagnet, self).__init__(B0)
