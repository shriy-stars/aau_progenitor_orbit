import sys, pickle, os, json
from importlib import reload
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

from multiprocessing import Pool
import multiprocessing as mp
mp.set_start_method("fork")
import time

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm

import agama 
agama.setNumThreads(1)
Gyr_to_AgamaTime = 1.0227 # 1 Gyr in Agama time units (kpc/(km/s))
import nbody_streams.agama_helper as ah

import emcee

from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import Galactocentric, ICRS, CartesianDifferential, CartesianRepresentation
from astropy import table


sys.path.append('../code/')
from coordinate_utils import get_rotation_matrix, sf_to_icrs, icrs_to_sf
from generate_sim_stream import create_stream_particle_spray
from stream_likelihood import log_likelihood, make_spline, log_prior, log_probability
from stream_data_utils import read_in_data


# Define file paths
BASE_POT_PATH = '../potential_files'
potMW_path = os.path.join(BASE_POT_PATH, 'McMillan17_nora.ini')
potLMC_path = os.path.join(BASE_POT_PATH, 'LMC_nora.ini')
accMW_path = os.path.join(BASE_POT_PATH, 'accMW')
trajLMC_path = os.path.join(BASE_POT_PATH, 'trajLMC')

## potential models to load
potMW = agama.Potential(file=potMW_path)
accMW = np.loadtxt(accMW_path)
trajLMC = np.loadtxt(trajLMC_path)
potacc  = agama.Potential(type='UniformAcceleration', file=accMW)
potLMC  = agama.Potential(file=potLMC_path)
potLMCm = agama.Potential(potential=potLMC, center=trajLMC)
potTotal= agama.Potential(potMW, potLMCm, potacc)

## defining our data
aau_member_path = '../data/aau_members.csv'
aau_distance_path = '../data/aau_bhb_rrl.csv'
aau_table = pd.read_csv(aau_member_path)
aau_bhb_rrl_data = pd.read_csv(aau_distance_path)
aau_bhb_rrl_data.columns = aau_bhb_rrl_data.columns.str.lower()
df_aau, prog_pars, prog_pars_icrs, df_distance_aau = read_in_data(aau_table, 'distance_modulus', aau_bhb_rrl_data)

## getting a initial first-guess from the median star in the data set

def select_prog_pars(df, df_distance):
    select_phi1 = (df['phi1'] > 1.0) & (df['phi1'] < 10.0)
    select_distance = (df_distance['distance_modulus'].notna())
    select_dist_phi1 = (df_distance['phi1'] > 1.0) & (df_distance['phi1'] < 10.0)

    phi1med = np.median(df.loc[select_phi1,'phi1'])
    phi2med = np.median(df.loc[select_phi1,'phi2'])
    distmed = np.median(df_distance.loc[select_dist_phi1 & select_distance, 'distance_modulus'])
    ramed = np.median(df.loc[select_phi1,'ra'])
    decmed = np.median(df.loc[select_phi1,'dec'])
    pmramed = np.median(df.loc[select_phi1, 'pmra'])
    pmdecmed = np.median(df.loc[select_phi1, 'pmdec'])
    rvmed = np.median(df.loc[select_phi1, 'vel_calib'])

    prog_pars = [float(phi1med), float(phi2med), float(distmed), float(pmramed), float(pmdecmed), float(rvmed)]
    prog_pars_icrs = [float(ramed), float(decmed), float(distmed), float(pmramed), float(pmdecmed), float(rvmed)]

    return prog_pars, prog_pars_icrs

prog_pars_aau, prog_pars_icrs_aau = select_prog_pars(df_aau, df_distance_aau)


## here is where we constrain the phi1 range to >-10.0 to only fit the Atlas region of AAU
select_phi1 = df_aau['phi1'].values > -10.0
select_phi1_dist = df_distance_aau['phi1'].values > -10.0

## making our data dictionary for the likelihood function
data_dict_aau = dict(
    phi1_obs               = df_aau['phi1'].values[select_phi1],
    phi1_obs_dist          = df_distance_aau['phi1'].values[select_phi1_dist],
    phi2_obs               = df_aau['phi2'].values[select_phi1],
    rv_obs                 = df_aau['vel_calib'].values[select_phi1],
    rv_obs_errors          = df_aau['vel_calib_std'].values[select_phi1],
    dist_obs               = (10**((df_distance_aau['distance_modulus'].values + 5) / 5) / 1000)[select_phi1_dist],
    dist_obs_errors        = (10**((df_distance_aau['distance_modulus'].values + 5) / 5) / 1000 * 0.1)[select_phi1_dist],
    pmra_cosdec_obs        = df_aau['pmra'].values[select_phi1],
    pmra_cosdec_obs_errors = df_aau['pmra_error'].values[select_phi1],
    pmdec_obs              = df_aau['pmdec'].values[select_phi1],
    pmdec_obs_errors       = df_aau['pmdec_error'].values[select_phi1],
)

## fixing the distance modulus artifact to just be distance for our initial median guess
prog_pars_aau[2] = 10**((prog_pars_aau[2] - 10) / 5)
prog_pars_icrs_aau[2] = 10**((prog_pars_icrs_aau[2] - 10) / 5)

phi1_pos = 0
prog_pars_aau[0] = phi1_pos #FIXING PHI1 @ specified location on stream-track
prog_pars_icrs_aau[0] = phi1_pos #this can be whatever because we aren't dealing with it

#Constructing stream progenitor present-day coordinates and frame
prog_mass, prog_scaleradius =  20_000, 10/1_000 # Msun, kpc
num_particles = 2_000
Age_stream_inGyr = 5.0
R = np.array(
[
            [0.83697865, 0.29481904, -0.4610298],
            [0.51616778,-0.70514011, 0.4861566],
            [0.18176238, 0.64487142, 0.74236331],
        ])

aau_rot_matrix = u.Quantity(R, unit=u.dimensionless_unscaled)
#aau_rot_matrix = get_rotation_matrix('AAU-ATLAS-L21', mws = mws) #optional galstreams stuff, but ignore here

'''
Piece de la resistance: the MCMC sampler

Uses emcee package with likelihood functions defined in stream_likelihood.py

likelihood function uses:
0.55 mas/yr for pm intrinsic dispersion from Li+2021
25 km/s for rv intrinsic dispersion from Li+2021

'''

nwalkers = 80
niter = 5000
nburnin = 500
ncpu = 80

#Taking a fraction of the prior range 
initial = prog_pars_aau[1:]
ndim = len(initial)
scales = np.array([0.01, 0.5, 0.01, 0.01, 1.0])  # phi2, dist, pmra, pmdec, rv
p0 = [np.array(initial) + scales * np.random.randn(ndim) for i in range(nwalkers)]
#p0 = [np.array(initial) + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]
OUTDIR = "mcmc_results"

data = (data_dict_aau, potTotal, prog_mass, prog_scaleradius, Age_stream_inGyr, num_particles, aau_rot_matrix)

def run_burnin(p0, nwalkers, ndim, log_probability, data, nburnin=nburnin, pool=None):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=data, pool=pool)
    p0, _, _ = sampler.run_mcmc(p0, nburnin, progress=False)
    return sampler, p0, np.mean(sampler.acceptance_fraction)

def run_production(p0, nwalkers, ndim, log_probability, data, niter=niter, pool=None,
                   h5=f"{OUTDIR}/production.h5", chunk=100, progress_cb=None):
    backend = emcee.backends.HDFBackend(h5)
    resuming = backend.iteration > 0
    if not resuming:
        backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=data,
                                    pool=pool, backend=backend)
    state = None if resuming else p0
    while backend.iteration < niter:
        n = min(chunk, niter - backend.iteration)
        state = sampler.run_mcmc(state, n, progress=False)
        if progress_cb is not None:
            progress_cb(backend.iteration, float(np.mean(sampler.acceptance_fraction)))
    return sampler


t_start = time.time()

## defining functions to generate summary files
def rtf_escape(s):
    return str(s).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

def write_summary(info, path=f"{OUTDIR}/summary.rtf"):
    """Rewrite the whole RTF summary from the `info` dict (ordered)."""
    lines = [r"{\rtf1\ansi\deff0{\fonttbl{\f0 Courier New;}}\f0\fs20",
             r"{\b AAU MCMC fit summary}\par\par"]
    for key, val in info.items():
        if isinstance(val, np.ndarray):
            val = np.array2string(val, precision=5)
        lines.append(rtf_escape(f"{key}: {val}") + r"\par")
    lines.append("}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


## writing out an output file
os.makedirs(OUTDIR, exist_ok=True)
burn_file = f"{OUTDIR}/burnin_best.npy"
param_names = ["phi2", "dist_kpc", "pmra_masyr", "pmdec_masyr", "rv_kms"]
info = {
    "Status": "burn-in running",
    "nwalkers": nwalkers, "ndim": ndim, "nburnin": nburnin,
    "niter (production steps)": niter, "CPUs used": ncpu,
    "Parameters": ", ".join(param_names),
}
write_summary(info)

## running our mcmc
with Pool(ncpu) as pool:
    if os.path.exists(burn_file):                      # restart: reuse cached burn-in
        best2 = np.load(burn_file)
        bz = np.load(f"{OUTDIR}/burnins.npz")
        acc1, acc2, best = float(bz["acc1"]), float(bz["acc2"]), bz["best1"]
    else:
        #first-burn in
        s1, _, acc1 = run_burnin(p0, nwalkers, ndim, log_probability, data, pool=pool)
        best = s1.get_chain(flat=True)[np.argmax(s1.get_log_prob(flat=True))]

        #run a second burn-in reinitialized with teh best results from burn-in
        p0_new = [best + scales * np.random.randn(ndim) for _ in range(nwalkers)]
        s2, _, acc2 = run_burnin(p0_new, nwalkers, ndim, log_probability, data, pool=pool)
        best2 = s2.get_chain(flat=True)[np.argmax(s2.get_log_prob(flat=True))]

        #saving our results, and use ipynb to plot results
        np.save(burn_file, best2)
        np.savez_compressed(f"{OUTDIR}/burnins.npz",
            chain1=s1.get_chain(), lp1=s1.get_log_prob(), acc1=acc1, best1=best,
            chain2=s2.get_chain(), lp2=s2.get_log_prob(), acc2=acc2, best2=best2)

    # writing it all out
    info["Burn-in 1 mean acceptance frac"] = f"{acc1:.3f}"
    info["Burn-in 1 best params"] = np.asarray(best)
    info["Burn-in 2 mean acceptance frac"] = f"{acc2:.3f}"
    info["Burn-in 2 best params"] = np.asarray(best2)
    info["Status"] = "production running"
    write_summary(info)

    def progress_cb(it, acc):
        info["Production iterations done"] = f"{it} / {niter}"
        info["Production mean acceptance so far"] = f"{acc:.3f}"
        info["Wall time this job (h)"] = f"{(time.time() - t_start) / 3600:.2f}"
        write_summary(info)

    #running production with best params from burnin
    p0_prod = [best2 + scales * np.random.randn(ndim) for _ in range(nwalkers)]
    sampler = run_production(p0_prod, nwalkers, ndim, log_probability, data,
                             pool=pool, progress_cb=progress_cb)

'''
Results r here below, i am saving them here to an npz file, and writing all the stuff into my sumamry file
'''
#pulling out our samples and chains from the mcmc 
flat = sampler.get_chain(flat=True)
flat_lp = sampler.get_log_prob(flat=True)
try:
    tau = sampler.get_autocorr_time(tol=0)
    info["Autocorrelation time per param"] = tau
    info["Chain length / tau (want > ~50)"] = np.round(niter / tau, 1)
except Exception as e:
    info["Autocorrelation time"] = f"failed: {e}"

info["Production mean acceptance (final)"] = f"{np.mean(sampler.acceptance_fraction):.3f}"
info["Max log-prob"] = f"{flat_lp.max():.3f}"
info["Max-likelihood params"] = flat[np.argmax(flat_lp)]
info["Wall time this job (h)"] = f"{(time.time() - t_start) / 3600:.2f}"
info["Status"] = "DONE"
write_summary(info)

np.savez_compressed(f"{OUTDIR}/aau_mcmc_samples.npz",
    chain=sampler.get_chain(), log_prob=sampler.get_log_prob(),
    acc_prod_per_walker=sampler.acceptance_fraction)

# # first burn-in
# sampler_burnin, p0_burnin, mean_acceptance_fraction_burnin = run_burnin(p0, nwalkers, ndim, log_probability, data)
# best = sampler_burnin.get_chain(flat=True)[np.argmax(sampler_burnin.get_log_prob(flat=True))]
# #print("Best params:", best)

# # reinitialize and run second burn-in
# p0_new = [best + scales * np.random.randn(ndim) for _ in range(nwalkers)]
# sampler_burnin2, p0_burnin2, mean_acceptance_fraction_burnin2 = run_burnin(p0_new, nwalkers, ndim, log_probability, data)
# best2 = sampler_burnin2.get_chain(flat=True)[np.argmax(sampler_burnin2.get_log_prob(flat=True))]
# #print("Best params after 2nd burn-in:", best2)

# # reinitialize around best and run production
# p0_final = best2
# p0_prod = [p0_final + scales * np.random.randn(ndim) for _ in range(nwalkers)]
# sampler, pos, prob, state = run_production(p0_prod, nwalkers, ndim, log_probability, data)