"""
Generate two appendix figures (Pareto and extrapolation) for the UNTRIMMED
(full-range) HMF data, with enlarged fonts matching Figure A1
(HMF_functions_untrimmed.pdf): axis labels 16, ticks 14, legend 14, annotations 14.

Outputs:
  Final_Plots/Pareto_HMF_untrimmed.pdf (+ Plots/...png)
  Final_Plots/extrapolation_HMF_untrimmed.pdf (+ Plots/...png)

Based on Parts 2 and 4 of trimmed_checks_and_plots.py, adapted for untrimmed data.
"""

import os
import re
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

os.chdir('/home/harry/Amelia_code')

# ── Font sizes matching Fig A1 ──────────────────────────────────────────────
FS_LABEL = 16
FS_TICK = 14
FS_LEG = 14
FS_ANNOT = 14

# ── Colour scheme ───────────────────────────────────────────────────────────
ESR_COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
LIT_COLOURS = {'P.Sch.': '#17becf', 'War.': '#bcbd22', 'Tin.': '#e377c2'}
LIT_STYLES = {'P.Sch.': '--', 'War.': '-.', 'Tin.': (0, (2, 2))}
LIT_LABELS = {'P.Sch.': 'Press-Schechter', 'War.': 'Warren', 'Tin.': 'Tinker'}


# ── Evaluation helpers ──────────────────────────────────────────────────────
def make_func(template):
    param_names = sorted(set(re.findall(r'a\d+', template)))

    def f(params, x):
        ns = {'x': x, 'pow': lambda a, b: np.float_power(np.abs(a), b),
              'Abs': np.abs, 'exp': np.exp,
              'log': lambda a: np.log(np.abs(a) + 1e-300),
              're': lambda z: np.real(z), 'im': lambda z: np.imag(z),
              'cos': np.cos}
        for name, val in zip(param_names, params):
            ns[name] = val
        with np.errstate(all='ignore'):
            result = eval(template, {"__builtins__": {}}, ns)
            return np.real(np.asarray(result, dtype=complex)).astype(float)
    return f, param_names


# ── Mass-variance relation (UNTRIMMED) ──────────────────────────────────────
logM_mvm, sigma_mvm, factor_mvm = np.loadtxt(
    'mass_variance_multiplier.txt.bak_untrimmed', dtype=float, unpack=True)
factor_of_sigma = interp1d(sigma_mvm, factor_mvm, kind='cubic', fill_value='extrapolate')
logM_of_sigma = interp1d(sigma_mvm, logM_mvm, kind='cubic', fill_value='extrapolate')
sigma_of_logM_mvm = interp1d(logM_mvm, sigma_mvm, kind='cubic', fill_value='extrapolate')
factor_of_logM_mvm = interp1d(logM_mvm, factor_mvm, kind='cubic', fill_value='extrapolate')

Veff = 1e9 / 0.6711**3
delta_logm = 0.2


def load_hmf_data_untrimmed(sim_id):
    data = np.loadtxt(f'hmf_files/hmf_{sim_id}_new.dat')
    n_full = len(data)
    factor = factor_mvm[:n_full]
    logM_bin = logM_mvm[:n_full]
    sigma = data[:, 0]
    counts = data[:, 1]
    logM = logM_bin + np.log10(0.6711)  # convert to h^-1 M_sun
    y = np.log10(counts / (Veff * delta_logm))
    y_err = 1 / (np.log(10) * np.sqrt(counts))
    return sigma, counts, logM, y, y_err, factor


def load_sim50_results_untrimmed():
    results = []
    with open('hmf_data/hmf_50_data/final_all.txt') as fh:
        for line in fh:
            parts = line.strip().split(';')
            if len(parts) < 5:
                continue
            template = parts[1]
            DL = float(parts[2])
            NLL = float(parts[3])
            pstr = parts[4].strip().strip('[]')
            params = np.array([float(v) for v in pstr.split()])
            results.append((template, DL, NLL, params))
    return results


# ── Literature fits on untrimmed data (sim 50) ──────────────────────────────
def press_schechter(x):
    delta_c = 1.686
    return np.sqrt(2.0 / np.pi) * (delta_c / x) * np.exp(-0.5 * (delta_c / x)**2)


def warren_func(x, params):
    a0, a1, a2, a3 = params
    return a0 * (np.power(x, a2) + a1) * np.exp(-a3 * np.power(x, -2.0))


def tinker_func(x, params):
    a0, a1, a2, a3 = params
    return a0 * (np.power(x / a2, -a1) + 1.0) * np.exp(-a3 * np.power(x, -2.0))


lit_params = {}
lit_sim50 = {}
with open('literature_fits_all_sims.txt') as fh:
    for line in fh:
        if line.startswith('#'):
            continue
        parts = line.strip().split(';')
        if int(parts[1]) == 50:
            name = parts[0]
            # File format: name;sim;DL;NLL;codelen;params — DL/NLL stored positive
            lit_sim50[name] = {'DL': float(parts[2]), 'NLL': float(parts[3])}
            if parts[5] != 'none':
                lit_params[name] = np.array([float(v) for v in parts[5].split()])


def eval_lit(name, sigma):
    if name == 'P.Sch.':
        return press_schechter(sigma)
    elif name == 'War.' and 'War.' in lit_params:
        return warren_func(sigma, lit_params['War.'])
    elif name == 'Tin.' and 'Tin.' in lit_params:
        return tinker_func(sigma, lit_params['Tin.'])
    return None


# ── ESR combined-DL ranking (untrimmed) ─────────────────────────────────────
esr_results = []
with open('hmf_combined_DL.txt') as fh:
    for line in fh:
        if line.startswith('#'):
            continue
        parts = line.strip().split(';')
        esr_results.append({
            'rank': int(parts[0]),
            'function': parts[1],
            'DL_combined': float(parts[2]),
            'sum_NLL': float(parts[4]),
        })

# Generation-complexity mapping (untrimmed)
searchcomp_map = {}
with open('hmf_func_gencomp.txt') as fh:
    for line in fh:
        if line.startswith('#'):
            continue
        parts = line.strip().split(';')
        if len(parts) >= 2 and parts[1] != '-1':
            searchcomp_map[parts[0]] = int(parts[1])

sim50_untrimmed = load_sim50_results_untrimmed()
sim50_dict = {t: p for (t, _, _, p) in sim50_untrimmed}
sim50_dl_dict = {t: dl for (t, dl, _, _) in sim50_untrimmed}
sim50_nll_dict = {t: nll for (t, _, nll, _) in sim50_untrimmed}


# ── PS-like detection ───────────────────────────────────────────────────────
def check_ps_like(func_str, params, sigma_vals=(100, 1000, 10000)):
    f_call, _ = make_func(func_str)
    f_vals, products = {}, {}
    for s in sigma_vals:
        try:
            fv = f_call(params, np.array([float(s)]))[0]
        except Exception:
            return False
        if not np.isfinite(fv) or fv <= 0:
            return False
        f_vals[s] = fv
        products[s] = s * fv
    prods = [products[s] for s in sigma_vals]
    if any(p <= 0 for p in prods):
        return False
    ratio_1, ratio_2 = prods[1] / prods[0], prods[2] / prods[1]
    converging = (0.1 < ratio_1 < 10) and (0.1 < ratio_2 < 10)
    f_decaying = f_vals[1000] < f_vals[100] and f_vals[10000] < f_vals[1000]
    if f_vals[100] > 0 and f_vals[1000] > 0:
        alpha_est = -np.log(f_vals[1000] / f_vals[100]) / np.log(10)
    else:
        return False
    return converging and f_decaying and np.isfinite(alpha_est) and (0.5 < alpha_est < 1.5)


# ══════════════════════════════════════════════════════════════════════════════
# Part A: Pareto front (untrimmed) — delegate to Pareto_plotter_neater
# ══════════════════════════════════════════════════════════════════════════════
print("=== Plot: Pareto HMF untrimmed ===")

from Pareto_plotter_neater import make_single_panel_figure, load_ps_like_for_hmf

hmf_ps_data = load_ps_like_for_hmf(sim=50)
n_ps = sum(1 for e in hmf_ps_data if e['ps_like'])
print(f"  Found {n_ps} PS-like functions")

os.makedirs('Plots/Old', exist_ok=True)
os.makedirs('Final_Plots', exist_ok=True)
make_single_panel_figure(
    'hmf_50',
    'Final_Plots/Pareto_HMF_untrimmed.pdf',
    ps_like_data=hmf_ps_data,
    label_fontsize=FS_LABEL,
    tick_fontsize=FS_TICK,
    legend_fontsize=FS_LEG,
    figsize=(6.8, 4.8),
    ylabel_x_offset=0.10,
)
print("Saved Pareto_HMF_untrimmed")


# ══════════════════════════════════════════════════════════════════════════════
# Part B: Extrapolation — two panels (untrimmed)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Plot: Extrapolation HMF untrimmed (2 panels) ===")

sigma_50, counts_50, logM_50, y_50, y_err_50, factor_50 = load_hmf_data_untrimmed(50)
top4_funcs = [(r['function'], f"ESR {i+1}") for i, r in enumerate(esr_results[:4])]

h_offset = np.log10(0.6711)
logM_eval_Msun = np.linspace(8, 20, 5000)
sigma_eval_logM = sigma_of_logM_mvm(logM_eval_Msun)
factor_eval_logM = factor_of_logM_mvm(logM_eval_Msun)
logM_eval_display = logM_eval_Msun + h_offset

logM_50_display = logM_50
sigma_eval_fine = np.geomspace(0.01, 20, 5000)

phi_data = 10**y_50
f_data_sigma = phi_data / factor_50
y_data_sigma = np.log10(f_data_sigma)

fig, (ax_logM, ax_sigma) = plt.subplots(1, 2, figsize=(14, 5.8))

# ---- Left: phi vs logM ----
ax_logM.errorbar(logM_50_display, y_50, yerr=y_err_50, fmt='x', color='black',
                 ms=6, elinewidth=0.8, capsize=0, zorder=10, label='Data')
ax_logM.axvspan(logM_50_display.min(), logM_50_display.max(),
                color='grey', alpha=0.08, zorder=0)

for idx, (func, label) in enumerate(top4_funcs):
    params = sim50_dict.get(func)
    if params is None:
        continue
    f_call, _ = make_func(func)
    f_vals = f_call(params, sigma_eval_logM)
    phi = f_vals * factor_eval_logM
    logy = np.where(phi > 0, np.log10(phi), -300.0)
    ax_logM.plot(logM_eval_display, logy, color=ESR_COLOURS[idx],
                 lw=1.5, label=label, zorder=5)

for name in ['P.Sch.', 'War.', 'Tin.']:
    f_lit = eval_lit(name, sigma_eval_logM)
    if f_lit is None:
        continue
    phi = f_lit * factor_eval_logM
    logy = np.where(phi > 0, np.log10(phi), -300.0)
    ax_logM.plot(logM_eval_display, logy, color=LIT_COLOURS[name],
                 ls=LIT_STYLES[name], lw=1.8, label=LIT_LABELS[name], zorder=4)

ax_logM.set_xlabel(r'$\log(M_h\,/\,h^{-1}M_\odot)$', fontsize=FS_LABEL)
ax_logM.set_ylabel(r'$\log\!\left(\phi\,/\,\mathrm{Mpc^{-3}\,dex^{-1}}\right)$', fontsize=FS_LABEL)
ax_logM.set_xlim(7.8, 16.8)
ax_logM.set_ylim(-12, 0)
ax_logM.tick_params(labelsize=FS_TICK)

# Inset
inset_logM = ax_logM.inset_axes([0.10, 0.05, 0.45, 0.45])
inset_xlim = (7.8, 12.3)
mask_d = (logM_50_display >= inset_xlim[0]) & (logM_50_display <= inset_xlim[1])
if mask_d.any():
    inset_logM.errorbar(logM_50_display[mask_d], y_50[mask_d], yerr=y_err_50[mask_d],
                        fmt='x', color='black', ms=5, elinewidth=0.6, capsize=0, zorder=10)
mask_e = (logM_eval_display >= inset_xlim[0]) & (logM_eval_display <= inset_xlim[1])
for idx, (func, label) in enumerate(top4_funcs):
    params = sim50_dict.get(func)
    if params is None:
        continue
    f_call, _ = make_func(func)
    f_vals = f_call(params, sigma_eval_logM)
    phi = f_vals * factor_eval_logM
    logy = np.where(phi > 0, np.log10(phi), -300.0)
    inset_logM.plot(logM_eval_display[mask_e], logy[mask_e],
                    color=ESR_COLOURS[idx], lw=1.3, zorder=5)
for name in ['P.Sch.', 'War.', 'Tin.']:
    f_lit = eval_lit(name, sigma_eval_logM)
    if f_lit is None:
        continue
    phi = f_lit * factor_eval_logM
    logy = np.where(phi > 0, np.log10(phi), -300.0)
    inset_logM.plot(logM_eval_display[mask_e], logy[mask_e],
                    color=LIT_COLOURS[name], ls=LIT_STYLES[name], lw=1.5, zorder=4)
inset_logM.set_xlim(*inset_xlim)
inset_logM.set_ylim(-3.5, -1)
inset_logM.tick_params(labelsize=FS_TICK - 2)
rect, connectors = ax_logM.indicate_inset_zoom(inset_logM, edgecolor='grey',
                                               alpha=0.5, linestyle='dotted')
for line in connectors:
    if line is not None:
        line.set_linestyle('dotted')

# ---- Right: f(sigma) vs sigma ----
ax_sigma.errorbar(sigma_50, y_data_sigma, yerr=y_err_50, fmt='x', color='black',
                  ms=6, elinewidth=0.8, capsize=0, zorder=10, label='Data')
ax_sigma.axvspan(sigma_50.min(), sigma_50.max(), color='grey', alpha=0.08, zorder=0)

for idx, (func, label) in enumerate(top4_funcs):
    params = sim50_dict.get(func)
    if params is None:
        continue
    f_call, _ = make_func(func)
    f_vals = f_call(params, sigma_eval_fine)
    logy = np.where(f_vals > 0, np.log10(f_vals), -300.0)
    ax_sigma.plot(sigma_eval_fine, logy, color=ESR_COLOURS[idx],
                  lw=1.5, label=label, zorder=5)

for name in ['P.Sch.', 'War.', 'Tin.']:
    f_lit = eval_lit(name, sigma_eval_fine)
    if f_lit is None:
        continue
    logy = np.where(f_lit > 0, np.log10(f_lit), -300.0)
    ax_sigma.plot(sigma_eval_fine, logy, color=LIT_COLOURS[name],
                  ls=LIT_STYLES[name], lw=1.8, label=LIT_LABELS[name], zorder=4)

ax_sigma.set_xlabel(r'$\sigma$', fontsize=FS_LABEL)
ax_sigma.set_ylabel(r'$\log\!\left(f(\sigma)\right)$', fontsize=FS_LABEL)
ax_sigma.yaxis.set_label_position('right')
ax_sigma.yaxis.tick_right()
ax_sigma.set_xlim(0, 8)
ax_sigma.set_ylim(-12, 2)
ax_sigma.tick_params(labelsize=FS_TICK)

# Joint legend across top
handles, labels = [], []
for ax in [ax_logM, ax_sigma]:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
fig.legend(handles, labels, loc='upper center', ncol=len(labels),
           fontsize=FS_LEG, frameon=True, bbox_to_anchor=(0.5, 1.10))

fig.tight_layout(w_pad=0)

plt.savefig('Plots/extrapolation_HMF_untrimmed.png', dpi=150, bbox_inches='tight')
plt.savefig('Final_Plots/extrapolation_HMF_untrimmed.pdf', dpi=200, bbox_inches='tight')
plt.close()
print("Saved extrapolation_HMF_untrimmed")

print("\nAll done!")
