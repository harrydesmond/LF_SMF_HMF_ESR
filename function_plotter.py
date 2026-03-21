"""Plot best-fit ESR and literature functions overlaid on LF, SMF, and HMF data.

Produces publication figures with three vertically stacked panels per dataset:
  - Top:    data points with best-fit curves (ESR, Schechter, Bernardi, etc.)
  - Middle: uncertainty-normalised residuals
  - Bottom: per-bin delta-NLL contributions relative to the best ESR function

Figures produced:
  - LF + SMF combined (Sersic and cmodel): Final_Plots/LF_SMF_functions.pdf
  - HMF (Quijote realisation 50):          Final_Plots/HMF_functions.pdf

Inputs:
    - *_final_functions.txt : function definitions with best-fit parameters
      (semicolon-delimited: source, complexity, DL, NLL, plot_fcn, blank_fcn)
    - *.txt data files      : binned LF/SMF/HMF data
    - mass_variance_multiplier.txt : sigma(M) and d(ln sigma)/d(log M) for HMF

Dependencies:
    numpy, matplotlib, pytexit, scipy
"""

import numpy as np
from matplotlib import pyplot as plt
from pytexit import py2tex
from scipy.special import gamma as Gamma
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import os
from scipy.stats import poisson

gamma = Gamma  # alias so eval() of Bernardi functions works
cm = plt.get_cmap('Set1')


def load_data(data_set):
    """Load data and function info for a given dataset."""
    source, comp, DL, NLL, plot_fcn, blank_fcn = np.loadtxt(
        '{}_final_functions.txt'.format(data_set), dtype=str, delimiter=';', unpack=True)
    DL, NLL = DL.astype(float), NLL.astype(float)

    if 'hmf' in data_set:
        _, counts, y_err, Veff_factor_delta, _ = np.loadtxt(
            'data/hmf_files/{}.dat'.format(data_set), dtype=float, delimiter=' ', unpack=True)
        logM, sigma, factor = np.loadtxt(
            'data/mass_variance_multiplier.txt', dtype=float, unpack=True)
        logM = logM[:len(counts)]
        sigma = sigma[:len(counts)]
        factor = factor[:len(counts)]
        x = sigma
        M = logM
        Veff = 1e9 / 0.6711**3
        delta_logm = 0.2
        y = np.log10(counts / (Veff * delta_logm))
        y_err = 1 / (np.log(10) * np.sqrt(counts))
    else:
        data_file = 'data/{}.txt'.format(data_set)
        if not os.path.isfile(data_file):
            # Some LF/SMF datasets use *_L / *_M names for fit outputs but keep
            # raw binned data in the base file (e.g., LF_cmodel.txt).
            for suffix in ('_L', '_M'):
                if data_set.endswith(suffix):
                    alt = 'data/{}.txt'.format(data_set[:-2])
                    if os.path.isfile(alt):
                        data_file = alt
                    break
        M_raw, y, y_err, Veff = np.loadtxt(
            data_file, dtype=float, delimiter=' ', unpack=True)
        if M_raw[0] < 0:
            # Column 0 is absolute magnitude; convert to L in L_sun
            M_sun_r = 4.67
            M_raw = 10**(-0.4 * (M_raw - M_sun_r))
        x = M_raw * 1e-09
        counts = 10**(y) * Veff
        M = np.log10(M_raw)
        factor = 1
        delta_logm = 1

    return {
        'source': source, 'comp': comp, 'DL': DL, 'NLL': NLL,
        'plot_fcn': plot_fcn, 'blank_fcn': blank_fcn,
        'x': x, 'M': M, 'y': y, 'y_err': y_err,
        'counts': counts, 'factor': factor, 'Veff': Veff,
        'delta_logm': delta_logm,
    }


def plot_dataset(data_set, colour, ax_data, ax_res, ax_nll, found, is_hmf=False, asymmetric_errors=True):
    """Plot data, residuals, and delta-NLL for a single dataset on given axes."""
    d = load_data(data_set)
    source, comp, plot_fcn, blank_fcn = d['source'], d['comp'], d['plot_fcn'], d['blank_fcn']
    x, M, y = d['x'], d['M'], d['y']
    counts, factor, Veff, delta_logm = d['counts'], d['factor'], d['Veff'], d['delta_logm']
    # Correct Poisson error on log10(phi): sigma = 1/(ln10 * sqrt(N))
    # (the file's y_err column is inflated by sqrt(10) for LF/SMF datasets)
    y_err = 1.0 / (np.log(10) * np.sqrt(np.maximum(counts, 1)))

    if is_hmf:
        x_label = '\sigma'
    elif 'LF' in data_set:
        x_label = 'L'
    else:
        x_label = 'M'

    if 'Ser' in data_set:
        ds_label = 'Sersic'
    elif 'cmodel' in data_set:
        ds_label = 'cmodel'
    else:
        ds_label = 'HMF'

    hmf_labels = {
        'P.Sch.': r'$f(\sigma) = \sqrt{\frac{2}{\pi}} \frac{\delta_c}{\sigma} \exp\!\left(-\frac{\delta_c^2}{2\sigma^2}\right)$',
        'War.': r'$f(\sigma) = \theta_1 (\sigma^{\theta_2} + \theta_3) \exp\!\left(-\frac{\theta_4}{\sigma^2}\right)$',
        'Tin.': r'$f(\sigma) = \theta_1\left[\left(\frac{\sigma}{\theta_2}\right)^{-\theta_3}\!\!+1\right]e^{-\theta_4/\sigma^2}$'
    }

    def _as_x_array(y_val):
        arr = np.asarray(y_val)
        if arr.ndim == 0:
            return np.full_like(x, float(arr), dtype=float)
        return arr

    # Find best ESR function for NLL reference (lowest DL = best)
    # Prefer ESR_T (top-ranked overall) if present; fall back to best-DL ESR
    esr_t_idx = [i for i, s in enumerate(source) if s == 'ESR_T']
    if esr_t_idx:
        best_esr_idx = esr_t_idx[np.argmin(d['DL'][esr_t_idx])]
    else:
        esr_idx = [i for i, s in enumerate(source) if s == 'ESR']
        best_esr_idx = esr_idx[np.argmin(d['DL'][esr_idx])]
    nll_old = None
    for idx, fcn in enumerate(plot_fcn):
        if idx == best_esr_idx:
            y_fcn = eval(fcn.replace("log", "np.log").replace("Abs", "abs").replace("exp", "np.exp"))
            y_fcn = _as_x_array(y_fcn)
            predicted = y_fcn * (factor * Veff * delta_logm)
            nll_old = predicted - counts * np.log(np.maximum(predicted, 1e-300))
            break

    for idx, fcn in enumerate(plot_fcn):
        y_fcn = eval(fcn.replace("log", "np.log").replace("Abs", "abs").replace("exp", "np.exp"))
        y_fcn = _as_x_array(y_fcn)
        predicted = y_fcn * (factor * Veff * delta_logm)
        nll_contributions = predicted - counts * np.log(np.maximum(predicted, 1e-300)) - nll_old

        if source[idx] == 'Sch.':
            y_plot = np.log10(y_fcn)
            lbl = 'Schechter' if source[idx] not in found else None
            if lbl:
                found.append(source[idx])
            ax_data.plot(M, y_plot, color=colour, linestyle='--', label=lbl)
            ax_res.plot(M, (y_plot - y) / y_err, color=colour, linestyle='--')
            ax_nll.plot(M, nll_contributions, color=colour, linestyle='--')

        elif source[idx] == 'P.Sch.':
            y_plot = np.log10(y_fcn * factor)
            ax_data.plot(M, y_plot, color='orange', linestyle='--', label='Press-Schechter')
            ax_res.plot(M, (y_plot - y) / y_err, color='orange', linestyle='--')
            ax_nll.plot(M, nll_contributions, color='orange', linestyle='--')

        elif source[idx] == 'War.':
            y_plot = np.log10(y_fcn * factor)
            ax_data.plot(M, y_plot, color='green', linestyle='--', label='Warren')
            ax_res.plot(M, (y_plot - y) / y_err, color='green', linestyle='--')
            ax_nll.plot(M, nll_contributions, color='green', linestyle='--')

        elif source[idx] == 'Tin.':
            y_plot = np.log10(y_fcn * factor)
            ax_data.plot(M, y_plot, color='blue', linestyle='--', label='Tinker')
            ax_res.plot(M, (y_plot - y) / y_err, color='blue', linestyle='--')
            ax_nll.plot(M, nll_contributions, color='blue', linestyle='--')

        elif idx == best_esr_idx:
            y_plot = np.log10(y_fcn * factor)
            lbl = 'ESR best' if 'ESR' not in found else None
            if lbl:
                found.append('ESR')
            ax_data.plot(M, y_plot, color=colour, label=lbl)
            ax_res.plot(M, (y_plot - y) / y_err, color=colour)
            # Skip delta-NLL for best ESR: identically zero by construction (not shown)

        elif 'Ber.' in source[idx]:
            y_plot = np.log10(y_fcn * factor)
            lbl = 'Bernardi' if source[idx] not in found else None
            if lbl:
                found.append(source[idx])
            ax_data.plot(M, y_plot, color=colour, linestyle=(0, (1, 1)), label=lbl)
            ax_res.plot(M, (y_plot - y) / y_err, color=colour, linestyle=(0, (1, 1)))
            ax_nll.plot(M, nll_contributions, color=colour, linestyle=(0, (1, 1)))

        else:
            continue

    # Data points
    ax_data.scatter(M, y, color=colour, marker='x', s=20, zorder=5)
    if asymmetric_errors:
        # Exact asymmetric Poisson 68% confidence intervals (16th and 84th percentiles)
        n = counts
        upper_count = poisson.ppf(0.8413, n)   # 84th percentile
        lower_count = poisson.ppf(0.1587, n)   # 16th percentile
        with np.errstate(divide='ignore', invalid='ignore'):
            y_upper = np.log10(upper_count / n)
            y_lower = np.log10(n / np.maximum(lower_count, 1e-300))
        y_upper = np.where(np.isfinite(y_upper) & (y_upper > 0), y_upper, y_err)
        y_lower = np.where(np.isfinite(y_lower) & (y_lower > 0) & (lower_count > 0), y_lower, y_err)
        ax_data.errorbar(x=M, y=y, yerr=[y_lower, y_upper], linestyle="none", color=colour, zorder=5)
    else:
        ax_data.errorbar(x=M, y=y, yerr=y_err, linestyle="none", color=colour, zorder=5)

    ax_res.axhline(0, color='grey', lw=0.5)
    ax_nll.axhline(0, color='grey', lw=0.5)

    ax_data.set_xlim([min(M) * 0.999, max(M) * 1.001])
    ax_res.set_xlim([min(M) * 0.999, max(M) * 1.001])
    ax_nll.set_xlim([min(M) * 0.999, max(M) * 1.001])


def overlay_best_models(data_set, colour, ax_data, ref_data_set='LF_Ser_L'):
    """Overlay best ESR/Schechter/Bernardi curves using a reference x-grid.

    Useful when a dataset has fitted functions available but no local raw
    plotting data file in the expected format.
    """
    source, comp, _, _, plot_fcn, _ = np.loadtxt(
        '{}_final_functions.txt'.format(data_set), dtype=str, delimiter=';', unpack=True)
    ref = load_data(ref_data_set)
    x = ref['x']
    M = ref['M']
    factor = ref['factor']

    def _as_x_array(y_val):
        arr = np.asarray(y_val)
        if arr.ndim == 0:
            return np.full_like(x, float(arr), dtype=float)
        return arr

    # Prefer ESR_T (top-ranked overall) if present; fall back to best-DL ESR
    DL_arr = np.array([float(d) for d in np.loadtxt(
        '{}_final_functions.txt'.format(data_set), dtype=str, delimiter=';', unpack=True)[2]])
    esr_t_idx = [i for i, s in enumerate(source) if s == 'ESR_T']
    if esr_t_idx:
        best_esr_idx = esr_t_idx[np.argmin(DL_arr[esr_t_idx])]
    else:
        esr_idx = [i for i, s in enumerate(source) if s == 'ESR']
        best_esr_idx = esr_idx[np.argmin(DL_arr[esr_idx])]

    for idx, fcn in enumerate(plot_fcn):
        src = source[idx]
        if not (idx == best_esr_idx or src == 'Sch.' or 'Ber.' in src):
            continue
        y_fcn = eval(fcn.replace("log", "np.log").replace("Abs", "abs").replace("exp", "np.exp"))
        y_fcn = _as_x_array(y_fcn)
        with np.errstate(divide='ignore', invalid='ignore'):
            y_plot = np.log10(y_fcn * factor)
        if src == 'Sch.':
            ax_data.plot(M, y_plot, color=colour, linestyle='--')
        elif 'Ber.' in src:
            ax_data.plot(M, y_plot, color=colour, linestyle=(0, (1, 1)))
        else:
            ax_data.plot(M, y_plot, color=colour)


# ============================================================
# Fig 1: LF + SMF combined (2 columns x 3 rows)
# ============================================================
fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(3, 2, hspace=0.0, wspace=0.05,
                          height_ratios=[3, 1, 1])

# LF column (left)
ax_lf_data = fig.add_subplot(outer[0, 0])
ax_lf_res = fig.add_subplot(outer[1, 0], sharex=ax_lf_data)
ax_lf_nll = fig.add_subplot(outer[2, 0], sharex=ax_lf_data)

# SMF column (right)
ax_smf_data = fig.add_subplot(outer[0, 1], sharey=ax_lf_data)
ax_smf_res = fig.add_subplot(outer[1, 1], sharex=ax_smf_data, sharey=ax_lf_res)
ax_smf_nll = fig.add_subplot(outer[2, 1], sharex=ax_smf_data, sharey=ax_lf_nll)

found = []
plot_dataset('LF_Ser_L', cm(0), ax_lf_data, ax_lf_res, ax_lf_nll, found)
plot_dataset('LF_cmodel_L', cm(1), ax_lf_data, ax_lf_res, ax_lf_nll, found)
plot_dataset('SMF_Ser_M', cm(0), ax_smf_data, ax_smf_res, ax_smf_nll, found)
plot_dataset('SMF_cmodel_M', cm(1), ax_smf_data, ax_smf_res, ax_smf_nll, found)

# Y-axis labels (left side)
ax_lf_data.set_ylabel(r'$\log\!\left(\phi / {\rm Mpc^{-3} \, dex^{-1}}\right)$', fontsize=12)
ax_lf_res.set_ylabel(r'$\frac{\rm Residual}{\rm Uncertainty}$', fontsize=12)
ax_lf_nll.set_ylabel(r'$\Delta$NLL', fontsize=12)

# Residual limits
ax_lf_res.set_ylim([-15, 15])
ax_smf_res.set_ylim([-15, 15])
ax_lf_nll.set_ylim([-50, 390])
ax_smf_nll.set_ylim([-50, 390])

# X-axis labels (bottom only)
ax_lf_nll.set_xlabel(r'$\log(L / L_\odot)$', fontsize=12)
ax_smf_nll.set_xlabel(r'$\log(M_\star / M_\odot)$', fontsize=12)

# Hide intermediate x-tick labels
plt.setp(ax_lf_data.get_xticklabels(), visible=False)
plt.setp(ax_lf_res.get_xticklabels(), visible=False)
plt.setp(ax_smf_data.get_xticklabels(), visible=False)
plt.setp(ax_smf_res.get_xticklabels(), visible=False)

# Hide y-tick labels on right column (shared y-axis)
plt.setp(ax_smf_data.get_yticklabels(), visible=False)
plt.setp(ax_smf_res.get_yticklabels(), visible=False)
plt.setp(ax_smf_nll.get_yticklabels(), visible=False)

# Show y-axis on right side of right column
ax_smf_data.yaxis.set_ticks_position('both')
ax_smf_res.yaxis.set_ticks_position('both')
ax_smf_nll.yaxis.set_ticks_position('both')
ax_smf_data.tick_params(right=True, labelright=False)
ax_smf_res.tick_params(right=True, labelright=False)
ax_smf_nll.tick_params(right=True, labelright=False)

# Tick label size
for ax in [ax_lf_data, ax_lf_res, ax_lf_nll, ax_smf_data, ax_smf_res, ax_smf_nll]:
    ax.tick_params(labelsize=11)

# Panel labels
ax_lf_data.set_title('Luminosity function', fontsize=13)
ax_smf_data.set_title('Stellar mass function', fontsize=13)

# Single legend across top
handles, labels = ax_lf_data.get_legend_handles_labels()
# Add linestyle legend items (generic, not coloured)
leg_handles = []
leg_labels = []
for h, l in zip(handles, labels):
    if l is not None and l not in leg_labels:
        if l == 'ESR best':
            leg_handles.append(Line2D([], [], color='black', linestyle='-'))
            leg_labels.append('ESR best')
        elif l == 'Schechter':
            leg_handles.append(Line2D([], [], color='black', linestyle='--'))
            leg_labels.append('Schechter')
        elif l == 'Bernardi':
            leg_handles.append(Line2D([], [], color='black', linestyle=(0, (1, 1))))
            leg_labels.append('Bernardi')

# Add dataset colour legend
leg_handles.append(Line2D([], [], color=cm(0), marker='x', linestyle='-', markersize=5))
leg_labels.append('Sersic')
leg_handles.append(Line2D([], [], color=cm(1), marker='x', linestyle='-', markersize=5))
leg_labels.append('cmodel')

fig.legend(leg_handles, leg_labels, loc='upper center', ncol=5, fontsize=11,
           bbox_to_anchor=(0.5, 0.985), frameon=True)

fig.subplots_adjust(top=0.90)

plt.savefig('Final_Plots/LF_SMF_functions.pdf', dpi=200, bbox_inches='tight')
plt.show()
plt.clf()


# ============================================================
# Fig 1 (symmetric errors): LF + SMF combined
# ============================================================
fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(3, 2, hspace=0.0, wspace=0.05,
                          height_ratios=[3, 1, 1])

ax_lf_data = fig.add_subplot(outer[0, 0])
ax_lf_res = fig.add_subplot(outer[1, 0], sharex=ax_lf_data)
ax_lf_nll = fig.add_subplot(outer[2, 0], sharex=ax_lf_data)
ax_smf_data = fig.add_subplot(outer[0, 1], sharey=ax_lf_data)
ax_smf_res = fig.add_subplot(outer[1, 1], sharex=ax_smf_data, sharey=ax_lf_res)
ax_smf_nll = fig.add_subplot(outer[2, 1], sharex=ax_smf_data, sharey=ax_lf_nll)

found = []
plot_dataset('LF_Ser_L', cm(0), ax_lf_data, ax_lf_res, ax_lf_nll, found, asymmetric_errors=False)
plot_dataset('LF_cmodel_L', cm(1), ax_lf_data, ax_lf_res, ax_lf_nll, found, asymmetric_errors=False)
plot_dataset('SMF_Ser_M', cm(0), ax_smf_data, ax_smf_res, ax_smf_nll, found, asymmetric_errors=False)
plot_dataset('SMF_cmodel_M', cm(1), ax_smf_data, ax_smf_res, ax_smf_nll, found, asymmetric_errors=False)

ax_lf_data.set_ylabel(r'$\log\!\left(\phi / {\rm Mpc^{-3} \, dex^{-1}}\right)$', fontsize=12)
ax_lf_res.set_ylabel(r'$\frac{\rm Residual}{\rm Uncertainty}$', fontsize=12)
ax_lf_nll.set_ylabel(r'$\Delta$NLL', fontsize=12)
ax_lf_res.set_ylim([-15, 15]); ax_smf_res.set_ylim([-15, 15])
ax_lf_nll.set_ylim([-50, 390]); ax_smf_nll.set_ylim([-50, 390])
ax_lf_nll.set_xlabel(r'$\log(L / L_\odot)$', fontsize=12)
ax_smf_nll.set_xlabel(r'$\log(M_\star / M_\odot)$', fontsize=12)
plt.setp(ax_lf_data.get_xticklabels(), visible=False)
plt.setp(ax_lf_res.get_xticklabels(), visible=False)
plt.setp(ax_smf_data.get_xticklabels(), visible=False)
plt.setp(ax_smf_res.get_xticklabels(), visible=False)
plt.setp(ax_smf_data.get_yticklabels(), visible=False)
plt.setp(ax_smf_res.get_yticklabels(), visible=False)
plt.setp(ax_smf_nll.get_yticklabels(), visible=False)
for a in [ax_smf_data, ax_smf_res, ax_smf_nll]:
    a.yaxis.set_ticks_position('both'); a.tick_params(right=True, labelright=False)
for a in [ax_lf_data, ax_lf_res, ax_lf_nll, ax_smf_data, ax_smf_res, ax_smf_nll]:
    a.tick_params(labelsize=11)
ax_lf_data.set_title('Luminosity function', fontsize=13)
ax_smf_data.set_title('Stellar mass function', fontsize=13)
handles, labels = ax_lf_data.get_legend_handles_labels()
leg_handles, leg_labels = [], []
for h, l in zip(handles, labels):
    if l is not None and l not in leg_labels:
        if l == 'ESR best':
            leg_handles.append(Line2D([], [], color='black', linestyle='-')); leg_labels.append('ESR best')
        elif l == 'Schechter':
            leg_handles.append(Line2D([], [], color='black', linestyle='--')); leg_labels.append('Schechter')
        elif l == 'Bernardi':
            leg_handles.append(Line2D([], [], color='black', linestyle=(0, (1, 1)))); leg_labels.append('Bernardi')
leg_handles.append(Line2D([], [], color=cm(0), marker='x', linestyle='-', markersize=5)); leg_labels.append('Sersic')
leg_handles.append(Line2D([], [], color=cm(1), marker='x', linestyle='-', markersize=5)); leg_labels.append('cmodel')
fig.legend(leg_handles, leg_labels, loc='upper center', ncol=5, fontsize=11, bbox_to_anchor=(0.5, 0.985), frameon=True)
fig.subplots_adjust(top=0.90)
plt.savefig('Final_Plots/LF_SMF_functions_symmetric.pdf', dpi=200, bbox_inches='tight')
plt.show()
plt.clf()


# ============================================================
# Fig 3: HMF (1 column x 3 rows)
# ============================================================
fig, (ax_data, ax_res, ax_nll) = plt.subplots(
    3, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [3, 1, 1]})

found = []
plot_dataset('hmf_50', cm(0), ax_data, ax_res, ax_nll, found, is_hmf=True)

ax_data.set_ylabel(r'$\log\!\left(\phi / {\rm Mpc^{-3} \, dex^{-1}}\right)$', fontsize=16)
ax_res.set_ylabel(r'$\frac{\rm Residual}{\rm Uncertainty}$', fontsize=16)
ax_nll.set_ylabel(r'$\Delta$NLL', fontsize=16)
ax_nll.set_xlabel(r'$\log(M_{\rm halo} / h^{-1} M_\odot)$', fontsize=16)

ax_res.set_ylim([-9, 9])
ax_nll.set_ylim([-50, 90])

plt.setp(ax_data.get_xticklabels(), visible=False)
plt.setp(ax_res.get_xticklabels(), visible=False)

for ax in [ax_data, ax_res, ax_nll]:
    ax.tick_params(labelsize=14)

# Legend
handles, labels = ax_data.get_legend_handles_labels()
ax_data.legend(handles, labels, fontsize=14)

fig.tight_layout()
fig.subplots_adjust(hspace=0)

plt.savefig('Final_Plots/HMF_functions.pdf', dpi=200, bbox_inches='tight')
plt.show()
plt.clf()


# ============================================================
# Fig 3 (symmetric errors): HMF
# ============================================================
fig, (ax_data, ax_res, ax_nll) = plt.subplots(
    3, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [3, 1, 1]})

found = []
plot_dataset('hmf_50', cm(0), ax_data, ax_res, ax_nll, found, is_hmf=True, asymmetric_errors=False)

ax_data.set_ylabel(r'$\log\!\left(\phi / {\rm Mpc^{-3} \, dex^{-1}}\right)$', fontsize=16)
ax_res.set_ylabel(r'$\frac{\rm Residual}{\rm Uncertainty}$', fontsize=16)
ax_nll.set_ylabel(r'$\Delta$NLL', fontsize=16)
ax_nll.set_xlabel(r'$\log(M_{\rm halo} / h^{-1} M_\odot)$', fontsize=16)
ax_res.set_ylim([-9, 9])
ax_nll.set_ylim([-50, 90])
plt.setp(ax_data.get_xticklabels(), visible=False)
plt.setp(ax_res.get_xticklabels(), visible=False)
for ax in [ax_data, ax_res, ax_nll]:
    ax.tick_params(labelsize=14)
handles, labels = ax_data.get_legend_handles_labels()
ax_data.legend(handles, labels, fontsize=14)
fig.tight_layout()
fig.subplots_adjust(hspace=0)
plt.savefig('Final_Plots/HMF_functions_symmetric.pdf', dpi=200, bbox_inches='tight')
plt.show()
plt.clf()
