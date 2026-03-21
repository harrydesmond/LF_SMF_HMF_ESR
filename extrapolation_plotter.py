"""Plot top 4 ESR functions + literature fits for LF Sersic, SMF Sersic,
SMF cmodel, and HMF, extending the x-range well beyond the data to reveal
extrapolation behaviour.

Usage:
    python3 extrapolation_plotter.py
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma as Gamma
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

gamma = Gamma  # alias so eval() of Bernardi functions works


# ── Data loading ──────────────────────────────────────────────────────────────

def load_lf_smf(data_set):
    """Return x (in 10^9 solar units), log10(phi), y_err, and log10(M_raw)."""
    data_file = 'data/{}.txt'.format(data_set)
    if not os.path.isfile(data_file):
        for suffix in ('_L', '_M'):
            if data_set.endswith(suffix):
                alt = 'data/{}.txt'.format(data_set[:-2])
                if os.path.isfile(alt):
                    data_file = alt
                break
    M_raw, y, y_err, Veff = np.loadtxt(
        data_file, dtype=float, unpack=True)
    if M_raw[0] < 0:
        # Column 0 is absolute magnitude; convert to L in L_sun
        M_sun_r = 4.67
        M_raw = 10**(-0.4 * (M_raw - M_sun_r))
    x = M_raw * 1e-9          # ESR variable  (L or M in units of 10^9)
    logM = np.log10(M_raw)    # for axis labelling
    return x, y, y_err, logM


def load_hmf(data_set):
    """Return x (=sigma), y (=log10 number density), y_err, and logM for HMF."""
    _, counts, y_err_raw, Veff_factor_delta, _ = np.loadtxt(
        'data/hmf_files/{}.dat'.format(data_set), dtype=float, unpack=True)
    logM, sigma, factor = np.loadtxt(
        'data/mass_variance_multiplier.txt', dtype=float, unpack=True)
    n = len(counts)
    logM, sigma = logM[:n], sigma[:n]
    x = sigma                                  # ESR variable
    Veff = 1e9 / 0.6711**3
    delta_logm = 0.2
    y = np.log10(counts / (Veff * delta_logm))  # log10(phi)
    y_err = 1 / (np.log(10) * np.sqrt(counts))
    return x, y, y_err, logM


def load_functions(data_set):
    """Load the *_final_functions.txt file and return per-function dicts."""
    source, comp, DL, NLL, plot_fcn, blank_fcn = np.loadtxt(
        '{}_final_functions.txt'.format(data_set),
        dtype=str, delimiter=';', unpack=True)
    DL = DL.astype(float)
    NLL = NLL.astype(float)
    return source, comp, DL, NLL, plot_fcn, blank_fcn


# ── Safe evaluation ───────────────────────────────────────────────────────────

def eval_fcn(fcn_str, x):
    """Evaluate a function string from the final_functions files."""
    # Replace pow first (before exp/log to avoid mangling np.exp → np.enp.power)
    safe = fcn_str
    safe = safe.replace("pow", "_POW_")
    safe = safe.replace("exp", "_EXP_")
    safe = safe.replace("log", "_LOG_")
    safe = safe.replace("Abs", "_ABS_")
    safe = safe.replace("abs", "_ABS_")
    safe = safe.replace("gamma", "_GAMMA_")
    safe = safe.replace("_POW_", "np.power")
    safe = safe.replace("_EXP_", "np.exp")
    safe = safe.replace("_LOG_", "np.log")
    safe = safe.replace("_ABS_", "np.abs")
    safe = safe.replace("_GAMMA_", "Gamma")
    # suppress warnings from e.g. log of negative numbers
    with np.errstate(all='ignore'):
        y = eval(safe)
    return np.where(np.isfinite(y), y, np.nan)


# ── Colour & style choices ────────────────────────────────────────────────────

ESR_COLOURS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']   # blue, orange, green, red
SCH_COLOUR   = '#9467bd'   # purple
BER_COLOUR   = '#8c564b'   # brown
DATA_COLOUR  = 'black'

LIT_COLOURS  = {'Sch': SCH_COLOUR, 'Ber': BER_COLOUR,
                'P.Sch.': '#17becf', 'War.': '#bcbd22', 'Tin.': '#e377c2',
                'Ber.': BER_COLOUR}
LIT_STYLES   = {'Sch': '--', 'Ber': (0, (2, 2)),
                'P.Sch.': '--', 'War.': '-.', 'Tin.': (0, (2, 2)),
                'Ber.': (0, (2, 2))}
LIT_LABELS   = {'Sch': 'Schechter', 'Ber': 'Bernardi',
                'P.Sch.': 'Press-Schechter', 'War.': 'Warren', 'Tin.': 'Tinker',
                'Ber.': 'Bernardi'}

ESR_STYLES   = ['-', '-', '-', '-']


# ── Single-panel plotting function ───────────────────────────────────────────

def plot_extrapolation(data_set, ax, title, xlabel, x_extrap_range,
                       is_hmf=False, inset_xlim=(7, 9), lit_keys=None,
                       plot_xlim=None):
    """
    Plot the top 4 ESR functions + literature fits on *ax*,
    with data points and an extended x range.

    Parameters
    ----------
    data_set : str        e.g. 'LF_Ser_L' or 'hmf_50'
    ax : matplotlib Axes
    title : str           panel label
    xlabel : str          x-axis label
    x_extrap_range : (lo, hi)
        Range of the ESR variable x for evaluation.
    is_hmf : bool
    inset_xlim : (lo, hi) for the inset x-axis (in logM units)
    lit_keys : list of str  prefixes of literature sources to plot
    """
    if lit_keys is None:
        lit_keys = ['Sch', 'Ber']

    # Load data
    if is_hmf:
        x_data, y_data, y_err, logM_data = load_hmf(data_set)
        # For HMF, sigma *decreases* with mass → need interpolation for
        # a smooth logM evaluation grid.
        from scipy.interpolate import interp1d
        logM_all, sigma_all, factor_all = np.loadtxt(
            'data/mass_variance_multiplier.txt', dtype=float, unpack=True)
        sigma_of_logM = interp1d(logM_all, sigma_all, kind='cubic',
                                  fill_value='extrapolate')
        factor_of_logM = interp1d(logM_all, factor_all, kind='cubic',
                                   fill_value='extrapolate')
        logM_eval = np.linspace(x_extrap_range[0], x_extrap_range[1], 5000)
        x_eval = sigma_of_logM(logM_eval)
        factor_eval = factor_of_logM(logM_eval)
    else:
        x_data, y_data, y_err, logM_data = load_lf_smf(data_set)
        x_eval = np.geomspace(x_extrap_range[0], x_extrap_range[1], 5000)
        logM_eval = np.log10(x_eval * 1e9)
        factor_eval = 1.0  # no factor needed for LF/SMF

    source, comp, DL, NLL, plot_fcn, blank_fcn = load_functions(data_set)

    # --- identify the 4 best ESR by DL (most negative = best) ---
    # Prefer ESR_T (top-ranked overall), then ESR_C (combined-ranked); fall back to ESR
    esr_t_mask = np.array([s == 'ESR_T' for s in source])
    esr_c_mask = np.array([s == 'ESR_C' for s in source])
    if esr_t_mask.any():
        esr_idx = np.where(esr_t_mask)[0]
    elif esr_c_mask.any():
        esr_idx = np.where(esr_c_mask)[0]
    else:
        esr_idx = np.where(np.array([s == 'ESR' for s in source]))[0]
    esr_sorted = esr_idx[np.argsort(DL[esr_idx])]
    top4_esr = esr_sorted[:min(4, len(esr_sorted))]

    # --- data points ---
    ax.errorbar(logM_data, y_data, yerr=y_err, fmt='x', color=DATA_COLOUR,
                ms=5, elinewidth=0.7, capsize=0, zorder=10, label='Data')

    # --- Shade the data domain ---
    ax.axvspan(logM_data.min(), logM_data.max(), color='grey', alpha=0.08, zorder=0)

    # --- Plot top 4 ESR ---
    for rank, idx in enumerate(top4_esr):
        y_eval = eval_fcn(plot_fcn[idx], x_eval) * factor_eval
        logy = np.where(y_eval > 0, np.log10(y_eval), -300.0)
        label = 'ESR {}'.format(rank + 1)
        ax.plot(logM_eval, logy, color=ESR_COLOURS[rank],
                linestyle=ESR_STYLES[rank], lw=1.4, label=label, zorder=5)

    # --- Literature functions ---
    lit_indices = {}  # key → index, for reuse in inset
    for key in lit_keys:
        idxs = np.where(np.array([s.startswith(key) or s == key
                                   for s in source]))[0]
        if len(idxs):
            idx = idxs[0]
            lit_indices[key] = idx
            y_eval = eval_fcn(plot_fcn[idx], x_eval) * factor_eval
            logy = np.where(y_eval > 0, np.log10(y_eval), -300.0)
            ax.plot(logM_eval, logy,
                    color=LIT_COLOURS[key], linestyle=LIT_STYLES[key],
                    lw=1.6, label=LIT_LABELS[key], zorder=4)

    ax.set_ylim(bottom=-100)
    if plot_xlim is not None:
        ax.set_xlim(*plot_xlim)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r'$\log\!\left(\phi\right)$', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=13,
            va='top', ha='left', fontweight='bold')

    # --- Inset ---
    inset = ax.inset_axes([0.10, 0.05, 0.45, 0.45])

    mask_d = (logM_data >= inset_xlim[0]) & (logM_data <= inset_xlim[1])
    if mask_d.any():
        inset.errorbar(logM_data[mask_d], y_data[mask_d], yerr=y_err[mask_d],
                        fmt='x', color=DATA_COLOUR, ms=4, elinewidth=0.5,
                        capsize=0, zorder=10)

    mask_e = (logM_eval >= inset_xlim[0]) & (logM_eval <= inset_xlim[1])

    for rank, idx in enumerate(top4_esr):
        y_eval = eval_fcn(plot_fcn[idx], x_eval) * factor_eval
        logy = np.where(y_eval > 0, np.log10(y_eval), -300.0)
        inset.plot(logM_eval[mask_e], logy[mask_e], color=ESR_COLOURS[rank],
                   linestyle=ESR_STYLES[rank], lw=1.2, zorder=5)

    for key, idx in lit_indices.items():
        y_eval = eval_fcn(plot_fcn[idx], x_eval) * factor_eval
        logy = np.where(y_eval > 0, np.log10(y_eval), -300.0)
        inset.plot(logM_eval[mask_e], logy[mask_e],
                   color=LIT_COLOURS[key], linestyle=LIT_STYLES[key],
                   lw=1.4, zorder=4)

    inset.set_xlim(*inset_xlim)
    inset.set_ylim(-3.5, -1)
    inset.tick_params(labelsize=9)
    inset.set_xlabel('')
    inset.set_ylabel('')
    rect, connectors = ax.indicate_inset_zoom(inset, edgecolor='grey', alpha=0.5,
                                               linestyle='dotted')
    for line in connectors:
        if line is not None:
            line.set_linestyle('dotted')


# ── Main figure ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    from extrapolation_HMF_sigma import plot_hmf_sigma

    fig, axes = plt.subplots(3, 2, figsize=(14, 17))

    # LF Sersic
    plot_extrapolation(
        'LF_Ser_L', axes[0, 0],
        title='LF: Sersic',
        xlabel=r'$\log(L\,/\,L_\odot)$',
        x_extrap_range=(1e-4, 1e8),
        lit_keys=['Sch', 'Ber'],
        plot_xlim=(5, 16),
        inset_xlim=(5, 9),
    )
    axes[0, 0].set_ylabel(r'$\log\!\left(\phi\,/\,\mathrm{Mpc^{-3}\,dex^{-1}}\right)$', fontsize=12)

    # Share y-axis for row 1
    axes[0, 1].sharey(axes[0, 0])

    # LF cmodel
    plot_extrapolation(
        'LF_cmodel_L', axes[0, 1],
        title='LF: cmodel',
        xlabel=r'$\log(L\,/\,L_\odot)$',
        x_extrap_range=(1e-4, 1e8),
        lit_keys=['Sch', 'Ber'],
        plot_xlim=(5, 16),
        inset_xlim=(5, 9),
    )
    axes[0, 1].set_ylabel('')
    plt.setp(axes[0, 1].get_yticklabels(), visible=False)

    # SMF Sersic
    plot_extrapolation(
        'SMF_Ser_M', axes[1, 0],
        title='SMF: Sersic',
        xlabel=r'$\log(M_\star\,/\,M_\odot)$',
        x_extrap_range=(1e-4, 1e9),
        lit_keys=['Sch', 'Ber'],
        plot_xlim=(5, 16),
        inset_xlim=(5, 9),
    )
    axes[1, 0].set_ylabel(r'$\log\!\left(\phi\,/\,\mathrm{Mpc^{-3}\,dex^{-1}}\right)$', fontsize=12)

    # Share y-axis for row 2
    axes[1, 1].sharey(axes[1, 0])

    # SMF cmodel
    plot_extrapolation(
        'SMF_cmodel_M', axes[1, 1],
        title='SMF: cmodel',
        xlabel=r'$\log(M_\star\,/\,M_\odot)$',
        x_extrap_range=(1e-4, 1e9),
        lit_keys=['Sch', 'Ber'],
        plot_xlim=(5, 16),
        inset_xlim=(5, 9),
    )
    plt.setp(axes[1, 1].get_yticklabels(), visible=False)
    axes[1, 1].set_ylabel('')

    # ── Row 3: HMF (independent y-axes with different labels) ──
    # HMF vs logM
    plot_extrapolation(
        'hmf_50', axes[2, 0],
        title='HMF',
        xlabel=r'$\log(M_h\,/\,M_\odot)$',
        x_extrap_range=(8, 20),
        is_hmf=True,
        inset_xlim=(8, 12.5),
        lit_keys=['P.Sch.', 'War.', 'Tin.', 'Ber.'],
        plot_xlim=(8, 17),
    )
    axes[2, 0].set_ylabel(r'$\log\!\left(\phi\,/\,\mathrm{Mpc^{-3}\,dex^{-1}}\right)$', fontsize=12)

    # HMF vs sigma
    plot_hmf_sigma(axes[2, 1])
    axes[2, 1].set_xlabel(r'$\sigma$', fontsize=12)
    axes[2, 1].set_ylabel(r'$\log\!\left(f(\sigma)\right)$', fontsize=12)
    axes[2, 1].yaxis.set_label_position('right')
    axes[2, 1].yaxis.tick_right()
    axes[2, 1].tick_params(labelsize=10)
    axes[2, 1].text(0.05, 0.95, r'HMF ($\sigma$)', transform=axes[2, 1].transAxes,
                    fontsize=13, va='top', ha='left', fontweight='bold')

    plt.tight_layout(w_pad=0)

    # Collect unique legend entries across all panels
    all_handles, all_labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in all_labels:
                all_handles.append(h)
                all_labels.append(l)
    fig.legend(all_handles, all_labels, loc='upper center',
               ncol=len(all_labels), fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 1.03))

    plt.savefig('Final_Plots/extrapolation_behaviour.pdf',
                dpi=200, bbox_inches='tight')
    plt.savefig('Final_Plots/extrapolation_behaviour.png',
                dpi=200, bbox_inches='tight')
    plt.show()
    print("Saved to Final_Plots/extrapolation_behaviour.pdf")
