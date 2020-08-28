import numpy as np
try:
    import scipy.fft as fftpack
except:
    import scipy.fftpack as fftpack


class Noise(object):
    def __init__(self, rng=None):
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.RandomState()

    def __call__(self, times, rng=None):
        return np.zeros_like(times)

class MixedNoise(Noise):
    def __init__(self, components=[], rng=None):
        super().__init__(rng)
        self.components = components

    def __call__(self, times, rng=None):
        if rng is None: rng = self.rng
        noise = np.zeros_like(times)
        for component in self.components:
            noise += component(times, rng)
        return noise

class WhiteNoise(Noise):
    def __init__(self, scale, rng=None):
        super().__init__(rng)
        self.scale = scale

    def __call__(self, times, rng=None):
        if rng is None: rng = self.rng
        return rng.normal(loc=0.0, scale=self.scale, size=len(times))

class FreqNoise(Noise):
    def __init__(self, power, scale, rng=None):
        super().__init__(rng)
        self.power = power
        self.scale = scale

    def __call__(self, times, rng=None):
        if rng is None: rng = self.rng
        N = len(times)
        white = rng.normal(loc=0.0, scale=self.scale, size=N)
        freq = fftpack.fftfreq(N, d=times[1]-times[0])
        fft  = fftpack.fft(white)
        fft[freq!=0] *= np.power(np.abs(freq[freq!=0]), self.power/2.)
        fft[freq==0] = 0
        noise = fftpack.ifft(fft)
        return np.real(noise)

class LinearDrift(Noise):
    def __init__(self, scale, rng=None):
        self.scale = scale

    def __call__(self, times, rng=None):
        return times*self.scale

class ExponentialDrift(Noise):
    def __init__(self, scale, time_scale, rng=None):
        self.scale = scale
        self.time_scale = time_scale

    def __call__(self, times, rng=None):
        return  self.scale*np.exp(-times/self.time_scale)
