# FID_Simulation
The repository contains scripts to generate simulations of pulsed Nuclear Magnetic Resonance (pNMR) free induction decay (FID) signals.

# Installation

This package is a pure python package and does not yet need a full installation.
Just make sure you have installed the dependencies listed below and all scripts
from this repository in a single folder.

Dependencies:
* numpy
* scipy (subpackages: fft and integrate)
* numericalunits

# Documentation

## Number of cells

To estimate the number of cells needed for a sufficient precise simulation we
first estimate other uncertainty scales:

If we perform a FFT on a 10 ms long time series with a sampling rate of 1 MSPS,
than the FFT bin width is:
* NSample = 10000 --> NFFT = NSample/2 = 5000
* bandwidth = 1MSPS / NFFT = 0.2 kHz
* 0.2 kHz/61.79MHz = 3.2 ppm
This results in an 3.2 ppm effect. Note that the resolution can be even better
than the bin width by taking the shape and neighboring bins into account.

We find for 10^5 cells, that the deviation in Delta B is below 10 ppb and thus
about a factor of a few 100 below the FFT bin width resolution.
