* Simulation
  * t_90
    * I am still not sure about the exact relation between current and t_90
    * Compare values from PP
  * absolute induced voltage / current in Coil
    * B/<B> or B/|B| ?
    * How to get induced current from dM(r)?
    * use equivalent principle for antennas: Delta I <-> Delta B
  * Circuit effects
    * LC circuit has bandpass filter characteristics
  * what about other components of the probe?
* Noise shape
  * estimate S/N from exp data
    * use area under peak in PSD
  * estimate type of noise from data
    * use Allen Std. deviation
    * some filtering in digitizer
    * compare pre pulse to post pulse noise
* Frequency extraction Methods
  * Fit local sin waves
  * implement other methods
  * Hilbert Method
    * see Ran's DocDB and compare
    * there is a ~50 kHz oscillation left in the Phase plots
    * effect of discrete hilbert transform
    * what if we only hilbert transform the first 2 ms
    * dependence on length of fit window (error estimation)
    * errors at datapoints
* Gradients
  * Understand double Peak structure
    * also seen by Ran in real data
    * Has T1 or T2 an influence?
  * PP measurement
    * get software working
    * Analyze
    *

* Coil -> implement multi layer coil
          Dimensions in Ran's "MUON G-2 NMR FREQUENCY EXTRACTION" Sec. 5.3 Table 1
* EMF ->  see Ran's "MUON G-2 NMR FREQUENCY EXTRACTION" Sec. 5.1 Eq. 36:
          potential energy U = - current * Phi_muC
          Phi_muC = vec(mu) dot_product vec(B_coil) / current

          --> units
            mu = A m²
            B = T
            current = A
            --> U = J or eV
            That corresponds for electrons with charge e a voltage in V.

* Implement Baseline
* Implement Distortion
* Double check Noise w.r.t. Ran
* Chi2 Fit  
  * implement unweighted
  * implement diagonal
  * implement COV white Noise
  * implement COV data
  * implement smoothing effects
  * implement uncertainty estimate
  * implement chi2 -pvalue
* implement noise fitting at beginning / end of waveform
* recursive zero-crossing
* coils with multi layers
* g(omega) histogram
* EMF
* Docu

Tests
* bias
* uncertainty / RMS
  * omega_fit - omega_true
  * omega_true - omega_center
  * vs fit window Length
  * vs polynomial order
