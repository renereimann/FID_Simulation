from FreeInductionDecay.units import *
from FreeInductionDecay.simulation.E989 import StorageRingMagnet, FixedProbe
from FreeInductionDecay.simulation.FID_sim import FID_simulation
from FreeInductionDecay.simulation.noise import FreqNoise, WhiteNoise
from FreeInductionDecay.analysis.phase_fit import PhaseFitFID, PhaseFitEcho
from FreeInductionDecay.analysis.hilbert_transform import HilbertTransform

import numpy as np
import matplotlib.pyplot as plt
import pickle

def run_spin_echo_simulation(lin_grad=0*ppm/cm, quad_grad=0*ppm/cm**2, N_cells=1000, N_ensamble=100, noise_scale=0.2*pc, seed=1, plotting=True, fit_window_scan=True, base_dir="./plots/Bloch_Echo", save_waveforms=True, **kwargs):
    # setup simulation
    b_field = StorageRingMagnet( )
    B0 = b_field.An[2]
    b_field.An[8] = lin_grad*B0
    b_field.An[15] = quad_grad*B0
    sim = FID_simulation(FixedProbe(), b_field, N_cells=N_cells, seed=seed)
    kwargs["probe"] = sim.probe
    grad_str = "%d_ppm_cm_%d_ppm_cm2"%(lin_grad/ppm*cm, quad_grad/ppm*cm**2)

    # simulation
    flux_raw, time = sim.spin_echo(pretrigger=True, time_pi=None, useBloch=True, pi_2_pulse_length=7.7*us) # None

    noise = WhiteNoise(scale=np.max(np.abs(flux_raw))*noise_scale)
    flux_noise = noise(time)
    flux = flux_raw + flux_noise

    # phase extraction
    hilb = HilbertTransform(time, flux)
    _, env = hilb.EnvelopeFunction()
    _, phase = hilb.PhaseFunction()

    if plotting:
        # frequency histogram
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
        plt.savefig("%s/g_spectrum_%s.png"%(base_dir, grad_str), dpi=200)

    if plotting:
        # FID signal
        plt.figure()
        plt.plot(time/ms, flux/uV, color="blue")
        plt.plot(time/ms, env/uV, color="k", ls="--")
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
        plt.savefig("%s/FID_Echo_with_envelope_%s.png"%(base_dir, grad_str), dpi=200)

    if plotting:
        thres = 3*uV
        t_start = time[:-1][np.logical_and(env[1:] >= thres, env[:-1]<thres)]
        t_end = time[:-1][np.logical_and(env[1:] < thres, env[:-1]>=thres)]
        # phase plot
        plt.figure()
        plt.plot(time/ms, phase, color="b")
        plt.xlim(0, 12)
        plt.text(0.5, 2, "$\pi$/2 pulse", rotation=90, horizontalalignment='left', verticalalignment='bottom')
        plt.text(4.1, 2, "$\pi$ pulse", rotation=90, horizontalalignment='left', verticalalignment='bottom')
        plt.text(7.9, 2, "Spin Echo", rotation=90, horizontalalignment='left', verticalalignment='bottom')
        for s, e in zip(t_start, t_end):
            plt.axvspan(s/ms, e/ms, color="gray", alpha=0.2)
        plt.axvline(sim.probe.time_pretrigger/ms, ls="--", color="k")
        plt.axvline(sim.probe.readout_length/ms, ls="--", color="k")
        plt.axvline((2*sim.probe.readout_length-sim.probe.time_pretrigger)/ms, ls="--", color="k")
        plt.xlabel("time in ms")
        plt.ylabel("phase in rad")
        plt.savefig("%s/Phase_function_%s.png"%(base_dir, grad_str), dpi=200)

    fit_fid = PhaseFitFID(**{"frac": 0.7, "tol": 1e-5, "window_size": 1/(50*kHz), "smoothing": True, "probe": sim.probe, "edge_ignore": 60*us})
    fit_echo = PhaseFitEcho(**{"frac": 0.7, "tol": 1e-5, "window_size": 1/(50*kHz), "smoothing": True, "probe": sim.probe})

    if fit_window_scan: # fit_window_scan
        f_scan = np.logspace(-0.5, 0.7, 20)
        fid_scatter = []
        for fit_window_fact in f_scan:
            fit_fid.fit_window_fact = fit_window_fact
            ensemble_FID = [fit_fid.fit(time, flux)/kHz for i in range(N_ensamble)]
            fid_scatter.append(np.std(ensemble_FID))
        echo_scatter = []
        for fit_window_fact in f_scan:
            fit_echo.fit_window_fact = fit_window_fact
            ensemble_Echo = [fit_echo.fit(time, flux)/kHz for i in range(N_ensamble)]
            echo_scatter.append(np.std(ensemble_Echo))

        fit_fid.fit_window_fact = f_scan[np.argmin(fid_scatter)]
        fit_echo.fit_window_fact = f_scan[np.argmin(echo_scatter)]

        if plotting:
            plt.figure()
            plt.plot(f_scan, 1e3*np.array(fid_scatter), label="FID", color="blue")
            plt.plot(f_scan, 1e3*np.array(echo_scatter), label="Echo", color="red")
            plt.axvline(fit_fid.fit_window_fact, color="blue")
            plt.axvline(fit_echo.fit_window_fact, color="red")
            plt.legend()
            plt.loglog()
            plt.grid()
            plt.xlabel("fit window factor")
            plt.ylabel("fitter convergence / Hz")
            plt.savefig("%s/fitter_accurancy_%s.png"%(base_dir, grad_str), dpi=200)

    true_f = sim.mean_frequency()/kHz
    # extraction plot

    print(fit_fid.fit(time, flux)/kHz)
    print(fit_echo.fit(time, flux)/kHz)
    print(true_f)
    plt.figure()
    fit_fid.plot()
    plt.savefig("%s/FID_frequency_extraction_%s.png"%(base_dir, grad_str), dpi=200, bbox_inches="tight")

    plt.figure()
    fit_echo.plot()
    plt.savefig("%s/Echo_frequency_extraction_%s.png"%(base_dir, grad_str), dpi=200, bbox_inches="tight")

    ensemble_FID = []
    ensemble_Echo = []
    ensamble_waveform = []
    for i in range(N_ensamble):
        N = noise(time)
        if save_waveforms:
            ensamble_waveform.append(flux_echo+N)
        ensemble_FID.append(fit_fid.fit(time, flux_raw+N)/kHz)
        ensemble_Echo.append(fit_echo.fit(time, flux_raw+N)/kHz)
    np.save("%s/waveforms_%s.npy"%(base_dir, grad_str), np.array(ensambe_waveform)/uV)

    bias_FID = np.mean(ensemble_FID)-true_f
    unce_FID = np.std(ensemble_FID)
    bias_Echo = np.mean(ensemble_Echo)-true_f
    unce_Echo = np.std(ensemble_Echo)
    print(bias_FID, unce_FID)
    print(bias_Echo, unce_Echo)
    print(unce_FID/unce_Echo)

    t_range_echo = fit_echo.t_range
    t_range_fid = fit_fid.t_range

    # save data
    res={"FID": {"f": fit_fid.fit_window_fact,
                 "t_range": (t_range_fid[0]/ms, t_range_fid[1]/ms),
                 "t_range_unit": "ms",
                 "bias": bias_FID,
                 "bias_unit": "kHz",
                 "uncertenty": unce_FID,
                 "uncertenty_unit": "kHz"},
         "Echo": {"f": fit_echo.fit_window_fact,
                  "t_range": (t_range_echo[0]/ms, t_range_echo[1]/ms),
                  "t_range_unit": "ms",
                  "bias": bias_Echo,
                  "bias_unit": "kHz",
                  "uncertenty": unce_Echo,
                  "uncertenty_unit": "kHz"}, }
    with open("%s/Freq_extraction_%s.pickle"%(base_dir, grad_str), "wb") as open_file:
        pickle.dump(res, open_file)

if __name__=="__main__":
    for lin_grad in np.linspace(0, 40, 20+1)*ppm/cm:
        try:
            run_spin_echo_simulation(lin_grad=lin_grad,
                                     quad_grad=0*ppm/cm**2,
                                     N_cells=1000,
                                     N_ensamble=100,
                                     noise_scale=0.2*pc,
                                     seed=1,
                                     frac=0.7,
                                     tol=1e-5,
                                     window_size=1/(50*kHz),
                                     smoothing=True,
                                     edge_ignore=60*us)
        except:
            continue
