#!/usr/bin/env python3
"""
param_uncertainties.py

Computes parameter uncertainties for the best-fit ESR and literature
functions quoted in Tables 1–3 of the paper.

Part A: Compute Fisher-matrix (Cramér-Rao) parameter uncertainties for
        representative HMF, LF and SMF functions using numdifftools.Hessian.
        sigma_i = 1/sqrt(H_ii) where H is the Hessian of the NLL.

Part B: Compare cross-simulation spread (16th–84th percentile) for HMF
        functions against the single-sim Fisher uncertainty.

Inputs:
  - data/hmf_files/hmf_50.dat
  - data/LF_Ser_L.txt, data/SMF_Ser_M.txt
  - LF_Ser_L_final_functions.txt, SMF_Ser_M_final_functions.txt
  - hmf_data/hmf_<sim>_data/final_all.txt  (100 sims)
  - literature_fits_all_sims.txt           (Warren/Tinker per-sim fits)

Outputs:
  - param_uncertainties_results.txt

Dependencies:
  numpy, numdifftools
"""

import numpy as np
import re
import os
import numdifftools as nd

BASE = '.'  # run from repo root

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_hmf_data(path):
    """Return sigma, count, norm from a 5-column HMF dat file."""
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1], data[:, 3]   # sigma, count, norm


def load_lf_smf_data(txt_path):
    """Return x (in units of 1e9 L_sun or M_sun), counts, Veff.
    Columns: raw_x, log10(phi), sigma, Veff
    raw_x may be in L_sun (positive) or magnitudes (negative).
    x = raw_x * 1e-9 (matching the ESR convention).
    counts = 10^log10(phi) * Veff
    """
    data = np.loadtxt(txt_path)
    raw_x  = data[:, 0]
    log_phi = data[:, 1]
    Veff   = data[:, 3]
    if raw_x[0] < 0:
        # Absolute magnitudes → convert to L_sun
        M_sun_r = 4.67
        raw_x = 10**(-0.4 * (raw_x - M_sun_r))
    x = raw_x * 1e-9
    counts = 10**log_phi * Veff
    return x, counts, Veff


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_func(template):
    """Return a callable f(params, x) from an ESR function template string."""
    param_names = sorted(set(re.findall(r'a\d+', template)))
    def f(params, x):
        ns = {
            'x': x,
            'pow': lambda a, b: np.float_power(np.abs(a), b),
            'Abs': np.abs,
            'exp': np.exp,
            'log': lambda a: np.log(np.abs(a)),
        }
        for name, val in zip(param_names, params):
            ns[name] = val
        with np.errstate(all='ignore'):
            return np.asarray(eval(template, {'__builtins__': {}}, ns), dtype=float)
    f.param_names = param_names
    return f


# ─────────────────────────────────────────────────────────────────────────────
# NLL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def hmf_nll_func(template, sigma, N, norm):
    """Return NLL(params) for HMF Poisson: sum(f(sigma)*norm - N*log(f(sigma)*norm))."""
    f = make_func(template)
    def nll(params):
        lam = f(params, sigma) * norm
        if not np.all(np.isfinite(lam)) or not np.all(lam > 0):
            return np.inf
        val = np.sum(lam - N * np.log(lam))
        return val if np.isfinite(val) else np.inf
    return nll


def lf_smf_nll_func(template, x, counts, Veff):
    """Return NLL(params) for LF/SMF Poisson: sum(f(x)*Veff - counts*log(f(x)*Veff))."""
    f = make_func(template)
    def nll(params):
        lam = f(params, x) * Veff
        if not np.all(np.isfinite(lam)) or not np.all(lam > 0):
            return np.inf
        val = np.sum(lam - counts * np.log(lam))
        return val if np.isfinite(val) else np.inf
    return nll


# ─────────────────────────────────────────────────────────────────────────────
# FISHER UNCERTAINTY
# ─────────────────────────────────────────────────────────────────────────────

def fisher_uncertainties(nll_fn, params_best, step_rel=1e-4):
    """Compute sigma_i = 1/sqrt(H_ii) using numdifftools.Hessian."""
    params_best = np.asarray(params_best, dtype=float)
    # Use relative step sizes; numdifftools accepts step as absolute or relative
    step = np.abs(params_best) * step_rel
    step[step < 1e-8] = 1e-8   # floor for near-zero params

    H_fn = nd.Hessian(nll_fn, step=step, method='central')
    H = H_fn(params_best)

    sigmas = np.full(len(params_best), np.nan)
    for i in range(len(params_best)):
        hii = H[i, i]
        if np.isfinite(hii) and hii > 0:
            sigmas[i] = 1.0 / np.sqrt(hii)
        elif np.isfinite(hii) and hii < 0:
            # Saddle or numerical issue – report as negative to flag
            sigmas[i] = -1.0 / np.sqrt(abs(hii))
    return sigmas, H


# ─────────────────────────────────────────────────────────────────────────────
# PART A: INDIVIDUAL FIT UNCERTAINTIES
# ─────────────────────────────────────────────────────────────────────────────

results = []
results.append('=' * 80)
results.append('PART A: Fisher-based parameter uncertainties (Cramér-Rao bound)')
results.append('=' * 80)

# ── HMF sim 50 data ───────────────────────────────────────────────────────────
hmf50_path = os.path.join(BASE, 'data', 'hmf_files', 'hmf_50.dat')
sigma50, N50, norm50 = load_hmf_data(hmf50_path)

# ── A1: HMF ESR best function ─────────────────────────────────────────────────
print('Computing A1: HMF ESR best function...')
template_esr = 'pow(Abs(a0),exp(pow(Abs(a1),(pow(Abs(a2 - x),a3)))))'
params_esr   = np.array([0.65734271, 1.70541354, 1.63275111, 2.78494662])

nll_esr = hmf_nll_func(template_esr, sigma50, N50, norm50)
nll_val_esr = nll_esr(params_esr)
sigmas_esr, H_esr = fisher_uncertainties(nll_esr, params_esr)

results.append('\nA1: HMF sim 50 — ESR best (comp 10)')
results.append(f'  Template : {template_esr}')
results.append(f'  NLL at best params: {nll_val_esr:.6f}')
results.append('  Parameters and Fisher uncertainties:')
pnames = ['a0', 'a1', 'a2', 'a3']
for i, (pn, pv, sig) in enumerate(zip(pnames, params_esr, sigmas_esr)):
    results.append(f'    {pn} = {pv:+.8f}  ±  {abs(sig):.2e}  (sigma_Fisher)')

# ── A2: HMF Warren ────────────────────────────────────────────────────────────
print('Computing A2: HMF Warren...')
template_war = 'a0*(pow(x,a2)+a1)*exp(-a3*pow(x,-2))'
params_war   = np.array([3.9544740027300525, -0.8514044830451555,
                          -0.08000126097738508, 0.810767329759749])

nll_war = hmf_nll_func(template_war, sigma50, N50, norm50)
nll_val_war = nll_war(params_war)
sigmas_war, H_war = fisher_uncertainties(nll_war, params_war)

results.append('\nA2: HMF sim 50 — Warren')
results.append(f'  Template : {template_war}')
results.append(f'  NLL at best params: {nll_val_war:.6f}')
results.append('  Parameters and Fisher uncertainties:')
war_names = ['a0', 'a1', 'a2', 'a3']
for pn, pv, sig in zip(war_names, params_war, sigmas_war):
    results.append(f'    {pn} = {pv:+.8f}  ±  {abs(sig):.2e}')

# ── A3: HMF Tinker ────────────────────────────────────────────────────────────
print('Computing A3: HMF Tinker...')
template_tin = 'a0*(pow(x/a2,-a1)+1)*exp(-a3*pow(x,-2))'
params_tin   = np.array([0.0023429021772495826, 0.7944154039239852,
                          1223.5821328359439, 0.9264052682953166])

nll_tin = hmf_nll_func(template_tin, sigma50, N50, norm50)
nll_val_tin = nll_tin(params_tin)
sigmas_tin, H_tin = fisher_uncertainties(nll_tin, params_tin)

results.append('\nA3: HMF sim 50 — Tinker')
results.append(f'  Template : {template_tin}')
results.append(f'  NLL at best params: {nll_val_tin:.6f}')
results.append('  Parameters and Fisher uncertainties:')
tin_names = ['a0', 'a1', 'a2', 'a3']
for pn, pv, sig in zip(tin_names, params_tin, sigmas_tin):
    results.append(f'    {pn} = {pv:+.8f}  ±  {abs(sig):.2e}')

# ── A4: LF Sérsic best ESR and Schechter ─────────────────────────────────────
print('Computing A4: LF Sérsic...')
x_lf, counts_lf, Veff_lf = load_lf_smf_data(os.path.join(BASE, 'data', 'LF_Ser_L.txt'))

results.append('\nA4: LF Sérsic')

# Parse best ESR from LF_Ser_L_final_functions.txt
lf_funcs = np.loadtxt(os.path.join(BASE, 'LF_Ser_L_final_functions.txt'),
                       dtype=str, delimiter=';', unpack=True)
lf_source, lf_comp, lf_DL, lf_NLL, lf_fcn, lf_blank = lf_funcs

# Best ESR: first ESR entry by DL (most negative)
esr_mask = np.array([s.startswith('ESR') for s in lf_source])
esr_DL = lf_DL[esr_mask].astype(float)
best_esr_idx = np.argmin(esr_DL)    # most negative = best
# Get from original indices
all_esr_idxs = np.where(esr_mask)[0]
best_esr_global = all_esr_idxs[best_esr_idx]

lf_esr_template = lf_blank[best_esr_global]
lf_esr_fcn      = lf_fcn[best_esr_global]
# Extract params from the numerical version of the function
# Parse param values from the fitted expression
lf_esr_param_names = sorted(set(re.findall(r'a\d+', lf_esr_template)))
# Extract numerical values from lf_esr_fcn by substituting template placeholders
# Build param list by eval-ing the fitted string at dummy x to find count
# Instead: extract numbers matching positions in the blank template
# Easier: use re to find all float numbers from the fitted function
# We'll use the template and function string to map values to param names
def extract_params_from_fitted(fitted_fcn, template, param_names):
    """Extract parameter values from a fitted function by comparing with template.

    Works correctly even when a param appears multiple times (e.g. Schechter a1).
    Maps each param to the value at its FIRST occurrence in the template.
    """
    float_re = r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'
    pattern = re.escape(template)
    # Replace every occurrence of each param name with a capture group.
    # Sort by length descending to avoid 'a1' matching inside 'a10'.
    for p in sorted(param_names, key=len, reverse=True):
        pattern = pattern.replace(re.escape(p), float_re)
    m = re.fullmatch(pattern, fitted_fcn)
    if m is None:
        return None
    groups = m.groups()
    # Determine which group index corresponds to each param (first occurrence).
    occurrence_order = re.findall(r'a\d+', template)   # with repetitions
    first_group_idx = {}
    for i, p in enumerate(occurrence_order):
        if p not in first_group_idx:
            first_group_idx[p] = i
    return [float(groups[first_group_idx[p]]) for p in param_names]

# For LF Sérsic ESR best
lf_esr_params = extract_params_from_fitted(lf_esr_fcn, lf_esr_template, lf_esr_param_names)
if lf_esr_params is None:
    # Fallback: extract all floats in order
    lf_esr_params = [float(v) for v in re.findall(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', lf_esr_fcn)
                     if re.match(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$', v)]
    lf_esr_params = lf_esr_params[:len(lf_esr_param_names)]

lf_esr_params = np.array(lf_esr_params)
nll_lf_esr = lf_smf_nll_func(lf_esr_template, x_lf, counts_lf, Veff_lf)
nll_val_lf_esr = nll_lf_esr(lf_esr_params)
sigmas_lf_esr, H_lf_esr = fisher_uncertainties(nll_lf_esr, lf_esr_params)

results.append(f'  Best ESR ({lf_source[best_esr_global]}, comp {lf_comp[best_esr_global]}):')
results.append(f'    Template : {lf_esr_template}')
results.append(f'    NLL at best params: {nll_val_lf_esr:.6f}')
results.append(f'    (File NLL: {float(lf_NLL[best_esr_global]):.6f})')
results.append('    Parameters and Fisher uncertainties:')
for pn, pv, sig in zip(lf_esr_param_names, lf_esr_params, sigmas_lf_esr):
    results.append(f'      {pn} = {pv:+.6e}  ±  {abs(sig):.2e}')

# LF Sérsic Schechter
sch_mask_lf = np.array([s == 'Sch.' for s in lf_source])
sch_idx_lf  = np.where(sch_mask_lf)[0][0]
lf_sch_template = lf_blank[sch_idx_lf]
lf_sch_fcn      = lf_fcn[sch_idx_lf]
lf_sch_param_names = sorted(set(re.findall(r'a\d+', lf_sch_template)))
lf_sch_params = extract_params_from_fitted(lf_sch_fcn, lf_sch_template, lf_sch_param_names)
if lf_sch_params is None:
    lf_sch_params = [float(v) for v in re.findall(r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?', lf_sch_fcn)][:len(lf_sch_param_names)]
lf_sch_params = np.array(lf_sch_params)

nll_lf_sch = lf_smf_nll_func(lf_sch_template, x_lf, counts_lf, Veff_lf)
nll_val_lf_sch = nll_lf_sch(lf_sch_params)
sigmas_lf_sch, H_lf_sch = fisher_uncertainties(nll_lf_sch, lf_sch_params)

results.append(f'\n  Schechter:')
results.append(f'    Template : {lf_sch_template}')
results.append(f'    NLL at best params: {nll_val_lf_sch:.6f}')
results.append(f'    (File NLL: {float(lf_NLL[sch_idx_lf]):.6f})')
results.append('    Parameters and Fisher uncertainties:')
for pn, pv, sig in zip(lf_sch_param_names, lf_sch_params, sigmas_lf_sch):
    results.append(f'      {pn} = {pv:+.6e}  ±  {abs(sig):.2e}')

# ── A5: SMF Sérsic best ESR and Schechter ────────────────────────────────────
print('Computing A5: SMF Sérsic...')
x_smf, counts_smf, Veff_smf = load_lf_smf_data(os.path.join(BASE, 'data', 'SMF_Ser_M.txt'))

results.append('\nA5: SMF Sérsic')

smf_funcs = np.loadtxt(os.path.join(BASE, 'SMF_Ser_M_final_functions.txt'),
                        dtype=str, delimiter=';', unpack=True)
smf_source, smf_comp, smf_DL, smf_NLL, smf_fcn, smf_blank = smf_funcs

# Best ESR
esr_mask_smf = np.array([s.startswith('ESR') for s in smf_source])
esr_DL_smf   = smf_DL[esr_mask_smf].astype(float)
best_esr_smf_idx = np.argmin(esr_DL_smf)
all_esr_smf_idxs = np.where(esr_mask_smf)[0]
best_esr_smf_global = all_esr_smf_idxs[best_esr_smf_idx]

smf_esr_template = smf_blank[best_esr_smf_global]
smf_esr_fcn      = smf_fcn[best_esr_smf_global]
smf_esr_param_names = sorted(set(re.findall(r'a\d+', smf_esr_template)))
smf_esr_params = extract_params_from_fitted(smf_esr_fcn, smf_esr_template, smf_esr_param_names)
if smf_esr_params is None:
    smf_esr_params = [float(v) for v in re.findall(r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?', smf_esr_fcn)][:len(smf_esr_param_names)]
smf_esr_params = np.array(smf_esr_params)

nll_smf_esr = lf_smf_nll_func(smf_esr_template, x_smf, counts_smf, Veff_smf)
nll_val_smf_esr = nll_smf_esr(smf_esr_params)
sigmas_smf_esr, H_smf_esr = fisher_uncertainties(nll_smf_esr, smf_esr_params)

results.append(f'  Best ESR ({smf_source[best_esr_smf_global]}, comp {smf_comp[best_esr_smf_global]}):')
results.append(f'    Template : {smf_esr_template}')
results.append(f'    NLL at best params: {nll_val_smf_esr:.6f}')
results.append(f'    (File NLL: {float(smf_NLL[best_esr_smf_global]):.6f})')
results.append('    Parameters and Fisher uncertainties:')
for pn, pv, sig in zip(smf_esr_param_names, smf_esr_params, sigmas_smf_esr):
    results.append(f'      {pn} = {pv:+.6e}  ±  {abs(sig):.2e}')

# SMF Schechter
sch_mask_smf = np.array([s == 'Sch.' for s in smf_source])
sch_idx_smf  = np.where(sch_mask_smf)[0][0]
smf_sch_template = smf_blank[sch_idx_smf]
smf_sch_fcn      = smf_fcn[sch_idx_smf]
smf_sch_param_names = sorted(set(re.findall(r'a\d+', smf_sch_template)))
smf_sch_params = extract_params_from_fitted(smf_sch_fcn, smf_sch_template, smf_sch_param_names)
if smf_sch_params is None:
    smf_sch_params = [float(v) for v in re.findall(r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?', smf_sch_fcn)][:len(smf_sch_param_names)]
smf_sch_params = np.array(smf_sch_params)

nll_smf_sch = lf_smf_nll_func(smf_sch_template, x_smf, counts_smf, Veff_smf)
nll_val_smf_sch = nll_smf_sch(smf_sch_params)
sigmas_smf_sch, H_smf_sch = fisher_uncertainties(nll_smf_sch, smf_sch_params)

results.append(f'\n  Schechter:')
results.append(f'    Template : {smf_sch_template}')
results.append(f'    NLL at best params: {nll_val_smf_sch:.6f}')
results.append(f'    (File NLL: {float(smf_NLL[sch_idx_smf]):.6f})')
results.append('    Parameters and Fisher uncertainties:')
for pn, pv, sig in zip(smf_sch_param_names, smf_sch_params, sigmas_smf_sch):
    results.append(f'      {pn} = {pv:+.6e}  ±  {abs(sig):.2e}')


# ─────────────────────────────────────────────────────────────────────────────
# PART B: CROSS-SIM SPREAD vs FISHER UNCERTAINTY (HMF)
# ─────────────────────────────────────────────────────────────────────────────

results.append('\n')
results.append('=' * 80)
results.append('PART B: Cross-simulation spread vs Fisher uncertainty (HMF, 100 sims)')
results.append('=' * 80)

# ── Load per-sim ESR parameters (best function) ───────────────────────────────
print('Loading per-sim ESR parameters...')
esr_template_hmf = 'pow(Abs(a0),exp(pow(Abs(a1),(pow(Abs(a2 - x),a3)))))'
esr_pnames = ['a0', 'a1', 'a2', 'a3']

esr_params_allsims = []
esr_sims_loaded = []
hmf_data_dir = os.path.join(BASE, 'hmf_data')
for sim_dir in sorted(os.listdir(hmf_data_dir)):
    if not sim_dir.startswith('hmf_') or not sim_dir.endswith('_data'):
        continue
    sim_num = sim_dir.replace('hmf_', '').replace('_data', '')
    fpath = os.path.join(hmf_data_dir, sim_dir, 'final_all.txt')
    if not os.path.exists(fpath):
        continue
    with open(fpath) as fh:
        for line in fh:
            if esr_template_hmf in line:
                parts = line.strip().split(';')
                # Format: rank;template;DL;NLL;[params array]
                if len(parts) < 5:
                    continue
                param_str = parts[4].strip()
                # param_str looks like "[0.65734271 1.70541354 ...]"
                param_str = param_str.strip('[]').split()
                if len(param_str) == 4:
                    p = np.array([float(v) for v in param_str])
                    # Handle sign degeneracy: a0 wrapped in Abs, a1 in Abs → take abs
                    p[0] = abs(p[0])
                    p[1] = abs(p[1])
                    esr_params_allsims.append(p)
                    esr_sims_loaded.append(sim_num)
                break

esr_params_allsims = np.array(esr_params_allsims)
print(f'  Loaded ESR params for {len(esr_params_allsims)} sims')

results.append(f'\nHMF ESR best function: {esr_template_hmf}')
results.append(f'(Loaded from {len(esr_params_allsims)} / 100 sims)')

# ── Load literature params across 100 sims ────────────────────────────────────
print('Loading literature fit parameters...')
lit_path = os.path.join(BASE, 'literature_fits_all_sims.txt')

war_params_all = []
tin_params_all = []

with open(lit_path) as fh:
    for line in fh:
        if line.startswith('#') or line.strip() == '':
            continue
        parts = line.strip().split(';')
        if len(parts) < 6:
            continue
        name = parts[0].strip()
        if name in ('War.', 'Tin.'):
            param_strs = parts[5].strip().split()
            params = np.array([float(v) for v in param_strs])
            if name == 'War.':
                war_params_all.append(params)
            elif name == 'Tin.':
                tin_params_all.append(params)

war_params_all = np.array(war_params_all)
tin_params_all = np.array(tin_params_all)
print(f'  Warren: {len(war_params_all)} sims, Tinker: {len(tin_params_all)} sims')

results.append(f'\nLiterature fits loaded: Warren ({len(war_params_all)} sims), '
               f'Tinker ({len(tin_params_all)} sims)')

# ── Fisher uncertainty from sim 50 (representative single sim) ────────────────
# Already computed above: sigmas_esr, sigmas_war, sigmas_tin

# ── Cross-sim 16th–84th percentile range ─────────────────────────────────────
def percentile_range(params_2d):
    """Return (p16, p84, range=p84-p16) for each parameter."""
    p16 = np.percentile(params_2d, 16, axis=0)
    p84 = np.percentile(params_2d, 84, axis=0)
    return p16, p84, p84 - p16

def ratio_report(name, pnames, params_2d, sigmas_fisher, template):
    """Return formatted lines comparing cross-sim spread to Fisher sigma."""
    lines = []
    lines.append(f'\nFunction: {name}')
    lines.append(f'  Template: {template}')
    lines.append(f'  N sims: {len(params_2d)}')
    lines.append(f'  {"Param":<8} {"Median":>14} {"p16":>14} {"p84":>14} '
                 f'{"(p84-p16)/2":>14} {"sigma_Fisher":>14} {"ratio":>8}')
    p16, p84, prange = percentile_range(params_2d)
    med = np.median(params_2d, axis=0)
    for i, pn in enumerate(pnames):
        sf = abs(sigmas_fisher[i]) if np.isfinite(sigmas_fisher[i]) else np.nan
        ratio = (prange[i] / 2.0) / sf if sf > 0 else np.nan
        lines.append(f'  {pn:<8} {med[i]:>14.4e} {p16[i]:>14.4e} {p84[i]:>14.4e} '
                     f'{prange[i]/2:>14.4e} {sf:>14.4e} {ratio:>8.3f}')
    lines.append(f'  Interpretation: ratio > 1 means cross-sim spread > Fisher sigma')
    return lines

results.extend(ratio_report(
    'HMF ESR (comp 10)', esr_pnames, esr_params_allsims, sigmas_esr, esr_template_hmf))

# Warren: handle sign convention (a0 always positive, a2 always negative in file)
# No Abs wrappers in Warren → no sign degeneracy to fix
results.extend(ratio_report(
    'HMF Warren', ['a0', 'a1', 'a2', 'a3'], war_params_all, sigmas_war, template_war))

results.extend(ratio_report(
    'HMF Tinker', ['a0', 'a1', 'a2', 'a3'], tin_params_all, sigmas_tin, template_tin))

# ── Summary table ─────────────────────────────────────────────────────────────
results.append('\n')
results.append('─' * 80)
results.append('SUMMARY: ratio (p84-p16)/(2*sigma_Fisher) per parameter')
results.append('─' * 80)
results.append('A ratio >> 1 means cosmic-variance / simulation-to-simulation scatter')
results.append('dominates over the single-simulation statistical uncertainty.')
results.append('A ratio ~ 1 means statistical noise and cosmic variance are comparable.')
results.append('A ratio << 1 means the Fisher (single-sim) uncertainty is generous.')

for name, pnames_loc, params_2d, sigmas_f in [
    ('ESR', esr_pnames, esr_params_allsims, sigmas_esr),
    ('Warren', ['a0', 'a1', 'a2', 'a3'], war_params_all, sigmas_war),
    ('Tinker', ['a0', 'a1', 'a2', 'a3'], tin_params_all, sigmas_tin),
]:
    p16, p84, prange = percentile_range(params_2d)
    results.append(f'\n  {name}:')
    for i, pn in enumerate(pnames_loc):
        sf = abs(sigmas_f[i]) if np.isfinite(sigmas_f[i]) else np.nan
        ratio = (prange[i] / 2.0) / sf if sf > 0 else np.nan
        results.append(f'    {pn}: (p84-p16)/2 = {prange[i]/2:.4e},  '
                       f'sigma_Fisher = {sf:.4e},  ratio = {ratio:.2f}')

# ─────────────────────────────────────────────────────────────────────────────
# WRITE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE, 'param_uncertainties_results.txt')
with open(out_path, 'w') as f:
    f.write('\n'.join(results) + '\n')

print(f'\nResults written to {out_path}')
# Also print to screen
print('\n'.join(results))
