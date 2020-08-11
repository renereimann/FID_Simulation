


def get_phase_from_hilbert(flux, window=None):
    N = len(flux)
    flux /= uV
    if window is not None:
        flux *= window(N)
    h = hilbert(flux)
    phi = np.arctan(np.imag(h)/np.real(h))
    oscillations = np.cumsum(np.pi*(np.sign(flux[1:]) != np.sign(flux[:-1])))
    phi += np.concatenate([[0], oscillations])

    plt.figure()
    plt.scatter(times/ms, np.degrees(phi- 314.16700919*times/ms)%180)
    plt.axhline(np.degrees(-1.57126946)%180, ls="--", color="k")
    plt.xlim([0.4, 0.6])
    plt.ylim(88, 92)

get_phase_from_hilbert(flux, window=signal.hamming)
