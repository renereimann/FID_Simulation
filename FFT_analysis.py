windowFunction = {"Hann": lambda nN: (np.sin(np.pi*nN))**2,
                  "Rectangular": lambda nN: 1,
                  "Triangular": lambda nN: 1-np.abs(2*nN - 1),
                  "Welch": lambda nN: 1 - (2*nN-1)**2,
                  "sine": lambda nN: np.sin(np.pi*nN),
                  "Blackman": lambda nN: 7938/18608 - 9240/18608 * np.cos(2*np.pi*nN) + 1430/18608*np.cos(4*np.pi*nN),
                  "Nuttal": lambda nN: 0.355768-0.487396*np.cos(2*np.pi*nN) +0.144232*np.cos(4*np.pi*nN) - 0.012604 *np.cos(6*np.pi*nN),
                  "Blackman-Nuttal": lambda nN: 0.3635819-0.4891775*np.cos(2*np.pi*nN) +0.1365995*np.cos(4*np.pi*nN) - 0.0106411 *np.cos(6*np.pi*nN),
                  }
