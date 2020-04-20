import numpy as np
import scipy.fft as fftpack

class Noise(object):
    r"""Class to generate noise and drift for time series. We plan to support
    different kind of noise and drift.
    """

    def __init__(self, white_noise=None, freq_power=None, scale_freq=None,
                 drift_lin=None,
                 drift_exp=None, drift_exp_time=None, rng=None):
        r"""Creates a Noise object that can be called to generate noise for time
        series.

        Parameters:
            - freq_power = Power of the f^alpha Power density spectrum. Default 1
            - drift_lin = strength of linear drift
            - drift_exp = strength of exponential dirft
            - rng = RandomState object used to generate random numbers.
        """
        self.freq_power = freq_power
        self.scale_freq = scale_freq
        self.white_noise = white_noise
        self.drift_lin = drift_lin
        self.drift_exp = drift_exp
        self.drift_exp_time = drift_exp_time
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.RandomState()

    def get_freq_noise(self, times, rng):
        N = len(times)
        rand_noise = rng.normal(loc=0.0, scale=self.scale_freq, size=N)
        freq = fftpack.fftfreq(N, d=times[1]-times[0])
        fft  = fftpack.fft(rand_noise)
        fft[freq!=0] *= np.power(np.abs(freq[freq!=0]), 0.5*self.freq_power)
        fft[freq==0] = 0
        noise = fftpack.ifft(fft)
        return np.real(noise)

    def get_white_noise(self, times, rng):
        return rng.normal(loc=0.0, scale=self.white_noise)

    def get_linear_drift(self, times):
        return times*self.drift_lin

    def get_exp_drift(self, times):
        return  self.drift_exp*np.exp(-times/self.drift_exp_time)

    def __call__(self, times, rng=None):
        if rng is None: rng = self.rng
        noise = np.zeros_like(times)
        if self.freq_power is not None and self.scale_freq is not None:
            noise += self.get_freq_noise(times, rng=rng)
        if self.white_noise is not None:
            noise += self.get_white_noise(times, rng=rng)
        if self.drift_lin is not None:
            noise += self.get_linear_drift(times)
        if self.drift_exp is not None and self.drift_exp_time is not None:
            noise += self.get_exp_drift(times)
        return noise
