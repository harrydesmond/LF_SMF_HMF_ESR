"""
Physicality checks + comparison plots for trimmed HMF ESR results.

Produces:
  1. Physicality check results for top 8 trimmed ESR functions
  2. Final_Plots/Pareto_HMF_trimmed.pdf — Pareto front with PS-like overlay
  3. Final_Plots/HMF_functions_trimmed.pdf — Function fits (phi vs logM, 3 panels)
  4. Final_Plots/extrapolation_HMF_trimmed.pdf — Extrapolation (phi vs logM)
"""

import os
import re
import sys
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

os.chdir('/home/harry/Amelia_code')

# ── Colour scheme (matching untrimmed) ──────────────────────────────────────
ESR_COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
LIT_COLOURS = {'P.Sch.': '#17becf', 'War.': '#bcbd22', 'Tin.': '#e377c2'}
LIT_STYLES  = {'P.Sch.': '--', 'War.': '-.', 'Tin.': (0, (2, 2))}
LIT_LABELS  = {'P.Sch.': 'Press-Schechter', 'War.': 'Warren', 'Tin.': 'Tinker'}

# ── Evaluation helpers ──────────────────────────────────────────────────────

def make_func(template):
    param_names = sorted(set(re.findall(r'a\d+', template)))
    def f(params, x):
        ns = {'x': x, 'pow': lambda a, b: np.float_power(np.abs(a), b),
              'Abs': np.abs, 'exp': np.exp, 'log': lambda a: np.log(np.abs(a) + 1e-300),
              're': lambda z: np.real(z), 'im': lambda z: np.imag(z), 'cos': np.cos}
        for name, val in zip(param_names, params):
            ns[name] = val
        with np.errstate(all='ignore'):
            result = eval(template, {"__builtins__": {}}, ns)
            return np.real(np.asarray(result, dtype=complex)).astype(float)
    return f, param_names


# ── Data loading ────────────────────────────────────────────────────────────

# Mass variance relation — use DIRECT per-bin values, not interpolation
logM_mvm, sigma_mvm, factor_mvm = np.loadtxt('mass_variance_multiplier.txt', dtype=float, unpack=True)
# For interpolation on fine grids (extrapolation plots), use sigma-based interp
factor_of_sigma = interp1d(sigma_mvm, factor_mvm, kind='cubic', fill_value='extrapolate')
logM_of_sigma = interp1d(sigma_mvm, logM_mvm, kind='cubic', fill_value='extrapolate')
sigma_of_logM_mvm = interp1d(logM_mvm, sigma_mvm, kind='cubic', fill_value='extrapolate')
factor_of_logM_mvm = interp1d(logM_mvm, factor_mvm, kind='cubic', fill_value='extrapolate')

Veff = 1e9 / 0.6711**3
delta_logm = 0.2


def load_hmf_data(sim_id, trimmed=False):
    data = np.loadtxt(f'hmf_files/hmf_{sim_id}_new.dat')
    n_full = len(data)
    if trimmed:
        data = data[2:]
        factor = factor_mvm[2:2+len(data)]
        logM_bin = logM_mvm[2:2+len(data)]
    else:
        factor = factor_mvm[:n_full]
        logM_bin = logM_mvm[:n_full]

    sigma = data[:, 0]
    counts = data[:, 1]
    logM = logM_bin + np.log10(0.6711)  # convert to h^-1 M_sun for x-axis
    y = np.log10(counts / (Veff * delta_logm))
    y_err = 1 / (np.log(10) * np.sqrt(counts))
    return sigma, counts, logM, y, y_err, factor


def load_trimmed_results(sim_id):
    results = []
    path = f'hmf_data/hmf_{sim_id}_data/final_all_trimmed.txt'
    with open(path) as fh:
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


# ── Literature functions ────────────────────────────────────────────────────

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
with open('literature_fits_trimmed.txt') as f:
    for line in f:
        if line.startswith('#'): continue
        parts = line.strip().split(';')
        if int(parts[1]) == 50:
            if parts[5] != 'none':
                lit_params[parts[0]] = np.array([float(v) for v in parts[5].split()])

lit_sim50 = {}
with open('literature_fits_trimmed.txt') as f:
    for line in f:
        if line.startswith('#'): continue
        parts = line.strip().split(';')
        if int(parts[1]) == 50:
            lit_sim50[parts[0]] = {'DL': -float(parts[2]), 'NLL': -float(parts[3])}


def eval_lit(name, sigma):
    if name == 'P.Sch.':
        return press_schechter(sigma)
    elif name == 'War.' and 'War.' in lit_params:
        return warren_func(sigma, lit_params['War.'])
    elif name == 'Tin.' and 'Tin.' in lit_params:
        return tinker_func(sigma, lit_params['Tin.'])
    return None


# ── Load ESR results ────────────────────────────────────────────────────────

esr_results = []
with open('hmf_combined_DL_trimmed_new.txt') as f:
    for line in f:
        if line.startswith('#'): continue
        parts = line.strip().split(';')
        esr_results.append({
            'rank': int(parts[0]),
            'function': parts[1],
            'DL_combined': float(parts[2]),
            'sum_NLL': float(parts[4]),
            'gencomp': int(parts[7]) if parts[7] != '-1' else None,
            'n_sims': int(parts[8]),
        })

# Load search complexity mapping
searchcomp_map = {}
with open('hmf_trimmed_searchcomp.txt') as f:
    for line in f:
        if line.startswith('#'): continue
        parts = line.strip().split(';')
        if len(parts) >= 2 and parts[1] != '-1':
            searchcomp_map[parts[0]] = int(parts[1])

sim50_trimmed = load_trimmed_results(50)
sim50_dict = {t: p for (t, _, _, p) in sim50_trimmed}
sim50_dl_dict = {t: dl for (t, dl, _, _) in sim50_trimmed}
sim50_nll_dict = {t: nll for (t, _, nll, _) in sim50_trimmed}

# ── PS-like detection ───────────────────────────────────────────────────────

def check_ps_like(func_str, params, sigma_vals=(100, 1000, 10000)):
    f_call, _ = make_func(func_str)
    f_vals, products = {}, {}
    for s in sigma_vals:
        try:
            fv = f_call(params, np.array([float(s)]))[0]
        except:
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
# Part 1: Physicality checks
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("PHYSICALITY CHECKS — Top 8 trimmed ESR functions")
print("=" * 80)

for i, r in enumerate(esr_results[:8]):
    func = r['function']
    params = sim50_dict.get(func)
    sc = searchcomp_map.get(func, '?')
    print(f"\n--- Rank {i+1} (search comp {sc}): {func} ---")
    if params is None:
        print("  WARNING: no sim 50 params found")
        continue
    f_callable, _ = make_func(func)

    pass_all = True
    # Check f→0 as σ→∞
    for factor in [10, 100, 1000]:
        y = f_callable(params, np.array([2.0 * factor]))
        if not np.isfinite(y[0]) or (abs(y[0]) > 1e-6 and factor == 1000):
            print(f"  ✗ f(σ) does not vanish at large σ")
            pass_all = False
            break
    else:
        print("  ✓ f(σ) → 0 as σ → ∞")

    # Check finite as σ→0
    for factor in [0.1, 0.01, 0.001]:
        y = f_callable(params, np.array([0.2 * factor]))
        if not np.isfinite(y[0]) or y[0] < -1e-10:
            print(f"  ✗ f(σ) not finite at small σ")
            pass_all = False
            break
    else:
        print("  ✓ f(σ) finite as σ → 0⁺")

    # Integrability
    x_int = np.logspace(-4, 4, 200000)
    y_int = f_callable(params, x_int)
    y_clean = np.where(np.isfinite(y_int), y_int, 0)
    integral = np.trapz(y_clean / x_int, x_int)
    if 0 < integral <= 1:
        print(f"  ✓ Mass fraction = {integral:.4f}")
    else:
        print(f"  ✗ Mass fraction = {integral:.4f}")
        pass_all = False

    print(f"  → {'PASSES' if pass_all else 'FAILS'}")


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Pareto front (search complexity, with PS-like)
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n=== Plot 2: Pareto HMF trimmed ===")

# Per SEARCH complexity best from sim 50
esr_comp_dl = {}
for comp in range(4, 11):
    path = f'hmf_trimmed_50_data/final_{comp}_trimmed.dat'
    if not os.path.exists(path):
        continue
    with open(path) as fh:
        line = fh.readline().strip()
    parts = line.split(';')
    esr_comp_dl[comp] = (float(parts[2]), float(parts[4]))

best_sim50_DL = min(dl for dl, nll in esr_comp_dl.values())
best_sim50_NLL = [nll for dl, nll in esr_comp_dl.values() if dl == best_sim50_DL][0]

# PS-like: use SEARCH complexity
print("  Detecting PS-like functions...")
ps_like_entries = []
for tmpl, dl50, nll50, p50 in sim50_trimmed:
    sc = searchcomp_map.get(tmpl)
    if sc is None or sc < 4 or sc > 10:
        continue
    if check_ps_like(tmpl, p50):
        ps_like_entries.append({'comp': sc, 'DL': dl50, 'NLL': nll50})
print(f"  Found {len(ps_like_entries)} PS-like functions")

comps_sorted = sorted(esr_comp_dl.keys())
dl_vals = [esr_comp_dl[c][0] - best_sim50_DL for c in comps_sorted]
nll_vals = [esr_comp_dl[c][1] - best_sim50_NLL for c in comps_sorted]

fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.plot(comps_sorted, dl_vals, 'o-', color='C0', ms=5, zorder=5, lw=1.5)
ax.plot(comps_sorted, nll_vals, 'o-', color='C3', ms=5, zorder=4, lw=1.5)

# PS-like Pareto front
if ps_like_entries:
    ps_by_comp_dl = defaultdict(list)
    ps_by_comp_nll = defaultdict(list)
    for e in ps_like_entries:
        ps_by_comp_dl[e['comp']].append(e['DL'] - best_sim50_DL)
        ps_by_comp_nll[e['comp']].append(e['NLL'] - best_sim50_NLL)
    ps_front_comps = sorted(ps_by_comp_dl.keys())
    ps_front_dl = [min(ps_by_comp_dl[c]) for c in ps_front_comps]
    ps_front_nll = [min(ps_by_comp_nll[c]) for c in ps_front_comps]
    ax.plot(ps_front_comps, ps_front_dl, 's-', color='purple', ms=6, zorder=7, lw=1.8)
    ax.plot(ps_front_comps, ps_front_nll, 's-', color='darkcyan', ms=6, zorder=7, lw=1.8)

# Literature
for name, comp, marker in [('War.', 14, '^'), ('Tin.', 16, 's')]:
    if name in lit_sim50:
        ax.scatter(comp, lit_sim50[name]['DL'] - best_sim50_DL, marker=marker,
                   color='C0', s=70, zorder=6, edgecolors='k', linewidths=0.5)
        ax.scatter(comp, lit_sim50[name]['NLL'] - best_sim50_NLL, marker=marker,
                   color='C3', s=70, zorder=6, edgecolors='k', linewidths=0.5)

if 'P.Sch.' in lit_sim50:
    ax.annotate(f'P.Sch.: $\\Delta$DL = {lit_sim50["P.Sch."]["DL"] - best_sim50_DL:.1e}',
                xy=(0.98, 0.98), xycoords='axes fraction', ha='right', va='top', fontsize=8, color='grey')

leg_handles = [
    Line2D([], [], color='C0', marker='o', linestyle='-', ms=5),
    Line2D([], [], color='C3', marker='o', linestyle='-', ms=5),
    Line2D([], [], color='purple', marker='s', linestyle='-', ms=5, lw=1.8),
    Line2D([], [], color='darkcyan', marker='s', linestyle='-', ms=5, lw=1.8),
    Line2D([], [], color='grey', marker='^', linestyle='None', ms=8, markeredgecolor='k'),
    Line2D([], [], color='grey', marker='s', linestyle='None', ms=8, markeredgecolor='k'),
]
leg_labels = [r'$\Delta$DL', r'$\Delta$NLL', r'PS-like $\Delta$DL', r'PS-like $\Delta$NLL',
              'Warren', 'Tinker']
ax.legend(leg_handles, leg_labels, fontsize=7.5, ncol=2, loc='upper left')
ax.set_xlabel('Complexity', fontsize=11)
ax.set_ylabel(r'$\Delta$DL / $\Delta$NLL', fontsize=11)
ax.tick_params(labelsize=10)
fig.tight_layout()

os.makedirs('Plots/Old', exist_ok=True)
plt.savefig('Plots/Pareto_HMF_trimmed.png', dpi=150)
plt.savefig('Final_Plots/Pareto_HMF_trimmed.pdf', bbox_inches='tight')
plt.close()
print("Saved Pareto_HMF_trimmed")


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Function fits — phi(logM) (matching Fig 5)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Plot 3: HMF functions (trimmed, sim 50) ===")

sigma_50, counts_50, logM_50, y_50, y_err_50, factor_50 = load_hmf_data(50, trimmed=True)
top4_funcs = [(r['function'], f"ESR {i+1}") for i, r in enumerate(esr_results[:4])]

fig, (ax_data, ax_res, ax_nll) = plt.subplots(
    3, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [3, 1, 1]})

ax_data.errorbar(logM_50, y_50, yerr=y_err_50, fmt='x', color='black',
                 ms=5, elinewidth=0.7, capsize=0, zorder=10, label='Data')

# ESR rank-1 reference for ΔNLL
f1_call, _ = make_func(esr_results[0]['function'])
f1_vals = f1_call(sim50_dict[esr_results[0]['function']], sigma_50)
lam1 = f1_vals * factor_50 * Veff * delta_logm
lam1 = np.where(lam1 > 0, lam1, 1e-300)
nll1_bins = lam1 - counts_50 * np.log(lam1)

for idx, (func, label) in enumerate(top4_funcs):
    params = sim50_dict.get(func)
    if params is None:
        continue
    f_call, _ = make_func(func)
    f_vals = f_call(params, sigma_50)
    phi_vals = f_vals * factor_50
    y_plot = np.log10(np.where(phi_vals > 0, phi_vals, 1e-300))
    ax_data.plot(logM_50, y_plot, color=ESR_COLOURS[idx], lw=1.4, label=label, zorder=5)
    ax_res.plot(logM_50, (y_plot - y_50) / y_err_50, color=ESR_COLOURS[idx], lw=0.8, marker='o', ms=3)
    if idx > 0:
        lam = f_vals * factor_50 * Veff * delta_logm
        lam = np.where(lam > 0, lam, 1e-300)
        ax_nll.plot(logM_50, (lam - counts_50 * np.log(lam)) - nll1_bins,
                    color=ESR_COLOURS[idx], lw=0.8, marker='o', ms=3)

for name in ['P.Sch.', 'War.', 'Tin.']:
    f_lit = eval_lit(name, sigma_50)
    if f_lit is None:
        continue
    phi_lit = f_lit * factor_50
    y_plot = np.log10(np.where(phi_lit > 0, phi_lit, 1e-300))
    ax_data.plot(logM_50, y_plot, color=LIT_COLOURS[name], ls=LIT_STYLES[name],
                 lw=1.6, label=LIT_LABELS[name], zorder=4)
    ax_res.plot(logM_50, (y_plot - y_50) / y_err_50, color=LIT_COLOURS[name],
                ls=LIT_STYLES[name], lw=0.8, marker='o', ms=3)
    lam = f_lit * factor_50 * Veff * delta_logm
    lam = np.where(lam > 0, lam, 1e-300)
    ax_nll.plot(logM_50, (lam - counts_50 * np.log(lam)) - nll1_bins,
                color=LIT_COLOURS[name], ls=LIT_STYLES[name], lw=0.8, marker='o', ms=3)

ax_data.set_ylabel(r'$\log\!\left(\phi / {\rm Mpc^{-3} \, dex^{-1}}\right)$', fontsize=16)
ax_res.set_ylabel(r'$\frac{\rm Residual}{\rm Uncertainty}$', fontsize=16)
ax_nll.set_ylabel(r'$\Delta$NLL', fontsize=16)
ax_nll.set_xlabel(r'$\log(M_{\rm halo} / h^{-1} M_\odot)$', fontsize=16)
ax_res.axhline(0, color='grey', ls=':', lw=0.5)
ax_nll.axhline(0, color='grey', ls=':', lw=0.5)
ax_res.set_ylim([-6, 6])
ax_nll.set_ylim([-3, 12])
plt.setp(ax_data.get_xticklabels(), visible=False)
plt.setp(ax_res.get_xticklabels(), visible=False)
for ax in [ax_data, ax_res, ax_nll]:
    ax.tick_params(labelsize=14)
ax_data.legend(fontsize=10)
fig.tight_layout()
fig.subplots_adjust(hspace=0)

plt.savefig('Plots/HMF_functions_trimmed.png', dpi=150, bbox_inches='tight')
plt.savefig('Final_Plots/HMF_functions_trimmed.pdf', dpi=200, bbox_inches='tight')
plt.close()
print("Saved HMF_functions_trimmed")


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Extrapolation — two panels matching Fig 4 HMF row
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Plot 4: Extrapolation HMF trimmed (2 panels) ===")

# --- Left panel: phi vs logM ---
# Evaluate in M_sun coords for sigma/factor lookup, then shift for h^-1 M_sun display
h_offset = np.log10(0.6711)  # -0.1732
logM_eval_Msun = np.linspace(8, 20, 5000)  # M_sun range for sigma lookup
sigma_eval_logM = sigma_of_logM_mvm(logM_eval_Msun)
factor_eval_logM = factor_of_logM_mvm(logM_eval_Msun)
logM_eval_display = logM_eval_Msun + h_offset  # convert to h^-1 M_sun for display

# Data in display coords
logM_50_display = logM_50  # already converted in load_hmf_data

# --- Right panel: f(sigma) vs sigma ---
sigma_eval_fine = np.geomspace(0.01, 20, 5000)

# Trimmed data in sigma space
phi_data = 10**y_50  # y_50 = log10(phi)
f_data_sigma = phi_data / factor_50
y_data_sigma = np.log10(f_data_sigma)

fig, (ax_logM, ax_sigma) = plt.subplots(1, 2, figsize=(14, 5.5))

# ---- Left: phi vs logM ----
ax_logM.errorbar(logM_50_display, y_50, yerr=y_err_50, fmt='x', color='black',
                 ms=5, elinewidth=0.7, capsize=0, zorder=10, label='Data (trimmed)')
ax_logM.axvspan(logM_50_display.min(), logM_50_display.max(), color='grey', alpha=0.08, zorder=0)

for idx, (func, label) in enumerate(top4_funcs):
    params = sim50_dict.get(func)
    if params is None: continue
    f_call, _ = make_func(func)
    f_vals = f_call(params, sigma_eval_logM)
    phi = f_vals * factor_eval_logM
    logy = np.where(phi > 0, np.log10(phi), -300.0)
    ax_logM.plot(logM_eval_display, logy, color=ESR_COLOURS[idx], lw=1.4, label=label, zorder=5)

for name in ['P.Sch.', 'War.', 'Tin.']:
    f_lit = eval_lit(name, sigma_eval_logM)
    if f_lit is None: continue
    phi = f_lit * factor_eval_logM
    logy = np.where(phi > 0, np.log10(phi), -300.0)
    ax_logM.plot(logM_eval_display, logy, color=LIT_COLOURS[name], ls=LIT_STYLES[name],
                 lw=1.6, label=LIT_LABELS[name], zorder=4)

ax_logM.set_xlabel(r'$\log(M_h\,/\,h^{-1}M_\odot)$', fontsize=12)
ax_logM.set_ylabel(r'$\log\!\left(\phi\,/\,\mathrm{Mpc^{-3}\,dex^{-1}}\right)$', fontsize=12)
ax_logM.set_xlim(7.8, 16.8)
ax_logM.set_ylim(-30, 0)
ax_logM.tick_params(labelsize=10)
ax_logM.text(0.05, 0.95, 'HMF (trimmed)', transform=ax_logM.transAxes,
             fontsize=13, va='top', ha='left', fontweight='bold')

# Inset for logM panel
inset_logM = ax_logM.inset_axes([0.10, 0.05, 0.45, 0.45])
inset_xlim = (7.8, 12.3)
mask_d = (logM_50_display >= inset_xlim[0]) & (logM_50_display <= inset_xlim[1])
if mask_d.any():
    inset_logM.errorbar(logM_50_display[mask_d], y_50[mask_d], yerr=y_err_50[mask_d],
                        fmt='x', color='black', ms=4, elinewidth=0.5, capsize=0, zorder=10)
mask_e = (logM_eval_display >= inset_xlim[0]) & (logM_eval_display <= inset_xlim[1])
for idx, (func, label) in enumerate(top4_funcs):
    params = sim50_dict.get(func)
    if params is None: continue
    f_call, _ = make_func(func)
    f_vals = f_call(params, sigma_eval_logM)
    phi = f_vals * factor_eval_logM
    logy = np.where(phi > 0, np.log10(phi), -300.0)
    inset_logM.plot(logM_eval_display[mask_e], logy[mask_e], color=ESR_COLOURS[idx], lw=1.2, zorder=5)
for name in ['P.Sch.', 'War.', 'Tin.']:
    f_lit = eval_lit(name, sigma_eval_logM)
    if f_lit is None: continue
    phi = f_lit * factor_eval_logM
    logy = np.where(phi > 0, np.log10(phi), -300.0)
    inset_logM.plot(logM_eval_display[mask_e], logy[mask_e], color=LIT_COLOURS[name],
                    ls=LIT_STYLES[name], lw=1.4, zorder=4)
inset_logM.set_xlim(*inset_xlim)
inset_logM.set_ylim(-3.5, -1)
inset_logM.tick_params(labelsize=9)
rect, connectors = ax_logM.indicate_inset_zoom(inset_logM, edgecolor='grey', alpha=0.5, linestyle='dotted')
for line in connectors:
    if line is not None:
        line.set_linestyle('dotted')

# ---- Right: f(sigma) vs sigma ----
ax_sigma.errorbar(sigma_50, y_data_sigma, yerr=y_err_50, fmt='x', color='black',
                  ms=5, elinewidth=0.7, capsize=0, zorder=10, label='Data (trimmed)')
ax_sigma.axvspan(sigma_50.min(), sigma_50.max(), color='grey', alpha=0.08, zorder=0)

for idx, (func, label) in enumerate(top4_funcs):
    params = sim50_dict.get(func)
    if params is None: continue
    f_call, _ = make_func(func)
    f_vals = f_call(params, sigma_eval_fine)
    logy = np.where(f_vals > 0, np.log10(f_vals), -300.0)
    ax_sigma.plot(sigma_eval_fine, logy, color=ESR_COLOURS[idx], lw=1.4, label=label, zorder=5)

for name in ['P.Sch.', 'War.', 'Tin.']:
    f_lit = eval_lit(name, sigma_eval_fine)
    if f_lit is None: continue
    logy = np.where(f_lit > 0, np.log10(f_lit), -300.0)
    ax_sigma.plot(sigma_eval_fine, logy, color=LIT_COLOURS[name], ls=LIT_STYLES[name],
                  lw=1.6, label=LIT_LABELS[name], zorder=4)

ax_sigma.set_xlabel(r'$\sigma$', fontsize=12)
ax_sigma.set_ylabel(r'$\log\!\left(f(\sigma)\right)$', fontsize=12)
ax_sigma.yaxis.set_label_position('right')
ax_sigma.yaxis.tick_right()
ax_sigma.set_xlim(0, 8)
ax_sigma.set_ylim(-7, 2)
ax_sigma.tick_params(labelsize=10)
ax_sigma.text(0.05, 0.95, r'HMF trimmed ($\sigma$)', transform=ax_sigma.transAxes,
              fontsize=13, va='top', ha='left', fontweight='bold')

# Legend across both panels
handles, labels = [], []
for ax in [ax_logM, ax_sigma]:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=10,
           frameon=True, bbox_to_anchor=(0.5, 1.04))

fig.tight_layout(w_pad=0)

plt.savefig('Plots/extrapolation_HMF_trimmed.png', dpi=150, bbox_inches='tight')
plt.savefig('Final_Plots/extrapolation_HMF_trimmed.pdf', dpi=200, bbox_inches='tight')
plt.close()
print("Saved extrapolation_HMF_trimmed")

print("\nAll done!")
