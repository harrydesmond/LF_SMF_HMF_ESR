"""Plot HMF extrapolation behaviour in terms of sigma (cf. Fig 6 bottom-right
which uses log10(M_halo)).

Usage:
    MPLBACKEND=Agg python3 extrapolation_HMF_sigma.py
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma as Gamma

gamma = Gamma


def eval_fcn(fcn_str, x):
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
    with np.errstate(all='ignore'):
        y = eval(safe)
    return np.where(np.isfinite(y), y, np.nan)


def plot_hmf_sigma(ax):
    """Plot HMF extrapolation in sigma-space on the given axes."""
    _, counts, _, _, _ = np.loadtxt('hmf_50_new.txt', dtype=float, unpack=True)
    logM, sigma, factor = np.loadtxt(
        'mass_variance_multiplier.txt', dtype=float, unpack=True)
    n = len(counts)
    sigma_data = sigma[:n]
    factor_data = factor[:n]
    Veff = 1e9 / 0.6711**3
    delta_logm = 0.2
    phi_data = counts / (Veff * delta_logm)
    f_data = phi_data / factor_data
    y_data = np.log10(f_data)
    y_err = 1 / (np.log(10) * np.sqrt(counts))

    source, comp, DL, NLL, plot_fcn, blank_fcn = np.loadtxt(
        'hmf_50_final_functions.txt', dtype=str, delimiter=';', unpack=True)
    DL = DL.astype(float)

    ESR_COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    LIT_COLOURS = {'P.Sch.': '#17becf', 'War.': '#bcbd22', 'Tin.': '#e377c2'}
    LIT_STYLES  = {'P.Sch.': '--', 'War.': '-.', 'Tin.': (0, (2, 2))}
    LIT_LABELS  = {'P.Sch.': 'Press-Schechter', 'War.': 'Warren', 'Tin.': 'Tinker'}

    sigma_eval = np.geomspace(0.01, 20, 5000)

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

    # Data
    ax.errorbar(sigma_data, y_data, yerr=y_err, fmt='x', color='black',
                ms=5, elinewidth=0.7, capsize=0, zorder=10, label='Data')
    ax.axvspan(sigma_data.min(), sigma_data.max(), color='grey', alpha=0.08, zorder=0)

    # ESR functions
    for rank, idx in enumerate(top4_esr):
        x = sigma_eval
        y_eval = eval_fcn(plot_fcn[idx], x)
        logy = np.where(y_eval > 0, np.log10(y_eval), -300.0)
        ax.plot(sigma_eval, logy, color=ESR_COLOURS[rank], lw=1.4,
                label='ESR {}'.format(rank + 1), zorder=5)

    # Literature functions
    for key in ['P.Sch.', 'War.', 'Tin.']:
        idxs = np.where(np.array([s == key for s in source]))[0]
        if len(idxs):
            idx = idxs[0]
            x = sigma_eval
            y_eval = eval_fcn(plot_fcn[idx], x)
            logy = np.where(y_eval > 0, np.log10(y_eval), -300.0)
            ax.plot(sigma_eval, logy, color=LIT_COLOURS[key],
                    linestyle=LIT_STYLES[key], lw=1.6,
                    label=LIT_LABELS[key], zorder=4)

    ax.set_xlim(0, 8)
    ax.set_ylim(-7, 0.5)


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_hmf_sigma(ax)
    ax.set_xlabel(r'$\sigma$', fontsize=14)
    ax.set_ylabel(r'$\log\!\left(f(\sigma)\right)$', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_title('HMF extrapolation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Final Plots/HMF_extrapolation_sigma.pdf', dpi=200, bbox_inches='tight')
    print("Saved Final Plots/HMF_extrapolation_sigma.pdf")
