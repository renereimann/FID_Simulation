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
