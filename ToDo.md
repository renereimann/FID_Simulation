Check:
* t_90 <-> current
* flux in coil: B/<B> or B/|B| ?
* pickup_flux/N_cells

ToDo:
* histogram omega_cell, done
* mean(B0), --> number of cells?, done
* check number of cells for simulations sufficient, done
* FFT as function of gradient strength, done
* Analytic calculation of mean B: Done, see overleaf
* 10^5 sim sufficient -> Delta B < 10 ppb << Fourier Limit
* 4 ppm an sens aus fourier limit
  * NSample= 10000 --> NFFT = NSample/2 = 5000
  * bandwidth = 1MSPS / NFFT = 0.2 kHz
  * 0.2 kHz/61.79MHz = 3.2 ppm

* Delta I <-> Delta B
* estimate S/N from exp data
  * use area under peak in PSD
* estimate type of noise from data, use Allen Std. deviation
  * Fit T^alpha instead
  * make FFT plot more meaningful
  * some filtering in digitizer
* Understand why Peakshape is how it is
  * what about T1 and T2
  * implement nummerical solution of FID
  * structure also seen by Ran in real data
* Analysis Methods
  * Fit local sin waves
  * Hilbert
    * see Ran's DocDB and compare
    * there is a ~50 kHz oscillation left in the Phase plots
    * effect of discrete hilbert transform
    * what if we only hilbert transform the first 2 ms
    * dependence on length of fit window (error estimation)
    * errors at datapoints
  * implement other methods
* H <-> M <-> B in Bloch Equation --> uses M, but only contains effects below few atomic distancies
* Frequency bandwidth implementation (10 µs --> 100 kHz), see Keeler soft and hard pulses
* Calculate effect of Magnetization B field
* implement Pulnging Probe

* that about motion / diffusion within material
* what about other components of the probe
* what about spin-spin interactions
