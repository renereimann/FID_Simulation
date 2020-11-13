from FreeInductionDecay.units import *
from FreeInductionDecay.simulation.E989 import StorageRingMagnet, FixedProbe
from FreeInductionDecay.simulation.FID_sim import FID_simulation
from FreeInductionDecay.simulation.noise import FreqNoise, WhiteNoise
from FreeInductionDecay.analysis.phase_fit import FID_analysis, Echo_analysis, fit_range_frac, fit_range
from FreeInductionDecay.analysis.hilbert_transform import HilbertTransform

import numpy as np
import matplotlib.pyplot as plt
import pickle

def run_spin_echo_simulation(lin_grad=0*ppm/cm, quad_grad=0*ppm/cm**2, N_cells=1000, N_ensamble=100, noise_scale=0.2*uV, seed=1, **kwargs):
        b_field = StorageRingMagnet( ) # 1,45 T
        B0 = b_field.An[2]
        b_field.An[8] = lin_grad*B0
        b_field.An[15] = quad_grad*B0
        sim = FID_simulation(FixedProbe(), b_field, N_cells=N_cells, seed=seed)
        noise = WhiteNoise(scale=noise_scale)
        grad_str = "%d_ppm_cm_%d_ppm_cm2"%(lin_grad/ppm*cm, quad_grad/ppm*cm**2)
        kwargs["probe"] = sim.probe

        freq, w = sim.frequency_spectrum()
        hi, edg = np.histogram(freq/kHz, bins=np.linspace((sim.mean_frequency()-3*sim.std_frequency())/kHz,(sim.mean_frequency()+3*sim.std_frequency())/kHz,31), weights=w)
        plt.figure()
        plt.step(edg, np.concatenate([hi, [0]]), where="post", label="cells in probe")
        plt.ylim(ymin=0)
        plt.axvline(sim.mean_frequency()/kHz, ls="--", color="k", label="Mean=%.2f kHz"%(sim.mean_frequency()/kHz))
        plt.axvline(sim.central_frequency()/kHz, ls=":", color="k", label="Central=%.2f kHz"%(sim.central_frequency()/kHz))
        plt.legend()
        plt.xlabel("frequency / kHz")
        plt.ylabel("# cells")
        true_f = sim.mean_frequency()/kHz
        plt.savefig("./plots/SpinEcho/g_spectrum_%s.png"%grad_str, dpi=200)

        flux_echo, time_echo = sim.spin_echo(pretrigger=True, noise=noise)
        hilb_echo = HilbertTransform(time_echo, flux_echo)
        _, env_echo = hilb_echo.EnvelopeFunction()
        _, phase_echo = hilb_echo.PhaseFunction()

        plt.figure()
        plt.plot(time_echo/ms, flux_echo/uV, color="blue")
        plt.plot(time_echo/ms, env_echo, color="k", ls="--")
        plt.xlim(0, 12)
        plt.ylim(-20, 20)
        plt.text(0.5, 2, "$\pi$/2 pulse", rotation=90, horizontalalignment='left', verticalalignment='bottom')
        plt.text(4.1, 2, "$\pi$ pulse", rotation=90, horizontalalignment='left', verticalalignment='bottom')
        plt.text(7.9, 2, "Spin Echo", rotation=90, horizontalalignment='left', verticalalignment='bottom')
        plt.axvline(sim.probe.time_pretrigger/ms, ls="--", color="k")
        plt.axvline(sim.probe.readout_length/ms, ls="--", color="k")
        plt.axvline((2*sim.probe.readout_length-sim.probe.time_pretrigger)/ms, ls="--", color="k")
        plt.xlabel("time in ms")
        plt.ylabel("Amplitude in a.u.")
        plt.savefig("plots/SpinEcho/FID_Echo_with_envelope_%s.png"%grad_str, dpi=200)

        plt.figure()
        plt.plot(time_echo/ms, flux_echo/uV, color="blue")
        plt.plot(time_echo/ms, env_echo, color="k", ls="--")
        plt.xlim(0, 12)
        plt.ylim(-20, 20)
        plt.text(0.5, 2, "$\pi$/2 pulse", rotation=90, horizontalalignment='left', verticalalignment='bottom')
        plt.text(4.1, 2, "$\pi$ pulse", rotation=90, horizontalalignment='left', verticalalignment='bottom')
        plt.text(7.9, 2, "Spin Echo", rotation=90, horizontalalignment='left', verticalalignment='bottom')
        plt.axvline(sim.probe.time_pretrigger/ms, ls="--", color="k")
        plt.axvline(sim.probe.readout_length/ms, ls="--", color="k")
        plt.axvline((2*sim.probe.readout_length-sim.probe.time_pretrigger)/ms, ls="--", color="k")
        plt.xlabel("time in ms")
        plt.ylabel("Amplitude in a.u.")
        plt.savefig("plots/SpinEcho/FID_Echo_with_envelope_%s.png"%grad_str, dpi=200)

        plt.figure()
        plt.plot(time_echo/ms, phase_echo, color="b")
        plt.xlim(0, 12)
        plt.text(0.5, 2, "$\pi$/2 pulse", rotation=90, horizontalalignment='left', verticalalignment='bottom')
        plt.text(4.1, 2, "$\pi$ pulse", rotation=90, horizontalalignment='left', verticalalignment='bottom')
        plt.text(7.9, 2, "Spin Echo", rotation=90, horizontalalignment='left', verticalalignment='bottom')
        plt.axvline(sim.probe.time_pretrigger/ms, ls="--", color="k")
        plt.axvline(sim.probe.readout_length/ms, ls="--", color="k")
        plt.axvline((2*sim.probe.readout_length-sim.probe.time_pretrigger)/ms, ls="--", color="k")
        plt.xlabel("time in ms")
        plt.ylabel("phase in rad")
        plt.savefig("plots/SpinEcho/Phase_function_%s.png"%grad_str, dpi=200)

        f_FID = 1.3
        f_Echo = 3

        if False:
            f_scan = np.logspace(-0.5, 0.7, 20)
            fid_scatter = []
            for fit_window_fact in f_scan:
                ensemble_FID = [FID_analysis(time_echo, flux_echo, fit_window_fact=fit_window_fact, **kwargs)/kHz for i in range(N_ensamble)]
                fid_scatter.append(np.std(ensemble_FID))
            echo_scatter = []
            for fit_window_fact in f_scan:
                ensemble_Echo = [Echo_analysis(time_echo, flux_echo, fit_window_fact=fit_window_fact, **kwargs)/kHz for i in range(N_ensamble)]
                echo_scatter.append(np.std(ensemble_Echo))

            f_FID = f_scan[np.argmin(fid_scatter)]
            f_Echo = f_scan[np.argmin(echo_scatter)]

            plt.figure()
            plt.plot(f_scan, 1e3*np.array(fid_scatter), label="FID", color="blue")
            plt.plot(f_scan, 1e3*np.array(echo_scatter), label="Echo", color="red")
            plt.axvline(f_FID, color="blue")
            plt.axvline(f_Echo, color="red")
            plt.legend()
            plt.loglog()
            plt.grid()
            plt.xlabel("fit window factor")
            plt.ylabel("fitter convergence / Hz")
            plt.savefig("plots/fitter_accurancy_%s.png"%grad_str, dpi=200)

        plt.figure()
        print(FID_analysis(time_echo, flux_echo, plotting=True, fit_window_fact=f_FID, **kwargs)/kHz)
        plt.savefig("plots/SpinEcho/FID_frequency_extraction_%s.png"%grad_str, dpi=200, bbox_inches="tight")
        plt.figure()
        print(Echo_analysis(time_echo, flux_echo, plotting=True, fit_window_fact=f_Echo, **kwargs)/kHz)
        print(true_f)
        plt.savefig("plots/SpinEcho/Echo_frequency_extraction_%s.png"%grad_str, dpi=200, bbox_inches="tight")

        ensemble_FID = []
        ensemble_Echo = []
        for i in range(N_ensamble):
            N = noise(time_echo)
            freq_FID = FID_analysis(time_echo, flux_echo+N,
                                fit_window_fact=f_FID, **kwargs)
            freq_echo = Echo_analysis(time_echo, flux_echo+N,
                                 fit_window_fact=f_Echo, **kwargs)
            ensemble_FID.append(freq_FID/kHz)
            ensemble_Echo.append(freq_echo/kHz)

        bias_FID = np.mean(ensemble_FID)-true_f
        unce_FID = np.std(ensemble_FID)
        bias_Echo = np.mean(ensemble_Echo)-true_f
        unce_Echo = np.std(ensemble_Echo)
        print(bias_FID, unce_FID)
        print(bias_Echo, unce_Echo)
        print(unce_FID/unce_Echo)

        t_range_echo = fit_range_frac(time_echo, env_echo, frac=kwargs["frac"], t0=2*kwargs["probe"].readout_length-kwargs["probe"].time_pretrigger)
        t_range_fid = fit_range(time_echo, env_echo, frac=kwargs["frac"], edge_ignore=kwargs["edge_ignore"],
                                pretrigger=kwargs["probe"].time_pretrigger, readout_length=kwargs["probe"].readout_length)

        res={"FID": {"f": f_FID,
                     "t_range": (t_range_fid[0]/ms, t_range_fid[1]/ms),
                     "t_range_unit": "ms",
                     "bias": bias_FID,
                     "bias_unit": "kHz",
                     "uncertenty": unce_FID,
                     "uncertenty_unit": "kHz"},
             "Echo": {"f": f_Echo,
                      "t_range": (t_range_echo[0]/ms, t_range_echo[1]/ms),
                      "t_range_unit": "ms",
                      "bias": bias_Echo,
                      "bias_unit": "kHz",
                      "uncertenty": unce_Echo,
                      "uncertenty_unit": "kHz"}, }
        with open("./data/SpinEcho/Freq_extraction_%s.pickle"%grad_str, "wb") as open_file:
            pickle.dump(res, open_file)

if __name__=="__main__":
    kwargs = {"frac": 0.7,
              "tol": 1e-5,
              "window_size": 1/(50*kHz),
              "smoothing": True,
              "edge_ignore": 60*us,
             }
    for lin_grad in np.linspace(24, 40, 8+1)*ppm/cm:
        try:
            run_spin_echo_simulation(lin_grad=lin_grad,
                                     quad_grad=0*ppm/cm**2,
                                     N_cells=1000,
                                     N_ensamble=100,
                                     noise_scale=0.2*uV,
                                     seed=1,
                                     **kwargs)
        except:
            continue
