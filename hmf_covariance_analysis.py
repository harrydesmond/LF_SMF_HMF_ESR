"""
Compute empirical covariance matrix across HMF sigma bins from 100 Quijote realisations.

Quantifies the size of off-diagonal correlations between sigma bins (used to
assess whether the Poisson likelihood — which assumes independent bins — is
appropriate for the HMF). Writes summary statistics, a correlation-matrix
heatmap, and (if non-diagonal) a Poisson-vs-Gaussian ranking comparison
against the top HMF functions in sim 50.

Usage:
    python3 hmf_covariance_analysis.py              # fiducial / restricted range (main text)
    python3 hmf_covariance_analysis.py --extended   # full range (appendix)

Inputs:
  - data/hmf_files/hmf_<sim>.dat (sim = 0..99; 5 cols). By default the 2
    lowest-mass bins (rows 0,1) are dropped on-the-fly; pass --extended to
    keep all bins.
  - hmf_50_final_functions.txt / hmf_50_final_functions_trimmed.txt
    (for the ranking-impact test)

Outputs:
  - hmf_covariance_results.txt / hmf_covariance_results_trimmed.txt
  - Final_Plots/hmf_correlation_matrix.pdf / _trimmed.pdf
"""

import argparse
import sys
# update this path to match your local ESR install
sys.path.insert(0, '/home/harry/Symbolic_regression/ESR-main/')

import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--extended', action='store_true',
                        help='Keep all bins (full-range analysis). Default: fiducial (drop 2 lowest-mass bins).')
    args = parser.parse_args()
    trimmed = not args.extended

    if trimmed:
        final_funcs_file = 'hmf_50_final_functions_trimmed.txt'
        results_outfile = 'hmf_covariance_results_trimmed.txt'
        plot_outfile = 'Final_Plots/hmf_correlation_matrix_trimmed.pdf'
        mode_label = 'TRIMMED / FIDUCIAL'
        N_TRIM = 2
    else:
        final_funcs_file = 'hmf_50_final_functions.txt'
        results_outfile = 'hmf_covariance_results.txt'
        plot_outfile = 'Final_Plots/hmf_correlation_matrix.pdf'
        mode_label = 'FULL RANGE'
        N_TRIM = 0

    # ─── 1. Load all 100 realisations ───

    data_dir = 'data/hmf_files'
    n_sims = 100

    all_data = []
    min_bins = 999
    for i in range(n_sims):
        d_full = np.loadtxt(os.path.join(data_dir, f'hmf_{i}.dat'))
        d = d_full[N_TRIM:] if N_TRIM else d_full
        all_data.append(d)
        if d.shape[0] < min_bins:
            min_bins = d.shape[0]

    print(f"Number of simulations: {n_sims}")
    trim_suffix = ' (fiducial / trimmed)' if trimmed else ''
    print(f"Minimum number of bins across sims{trim_suffix}: {min_bins}")

    n_bins = min_bins
    sigma_ref = all_data[0][:n_bins, 0]

    for i in range(n_sims):
        assert np.allclose(all_data[i][:n_bins, 0], sigma_ref), f"Sigma mismatch in sim {i}"

    counts_matrix = np.zeros((n_sims, n_bins))
    norms_matrix = np.zeros((n_sims, n_bins))
    for i in range(n_sims):
        counts_matrix[i, :] = all_data[i][:n_bins, 1]
        norms_matrix[i, :] = all_data[i][:n_bins, 3]

    norm_ref = norms_matrix[0]
    norms_identical = np.allclose(norms_matrix, norm_ref[np.newaxis, :], rtol=1e-10)
    print(f"Normalizations identical across sims: {norms_identical}")
    print(f"Sigma range: [{sigma_ref[-1]:.4f}, {sigma_ref[0]:.4f}]")
    print(f"Count range: [{counts_matrix.min():.0f}, {counts_matrix.max():.0f}]")
    print()

    # ─── 2. Covariance and correlation matrices ───

    mean_counts = np.mean(counts_matrix, axis=0)
    cov_matrix = np.cov(counts_matrix, rowvar=False, ddof=1)
    var_diag = np.diag(cov_matrix)
    std_diag = np.sqrt(var_diag)
    corr_matrix = cov_matrix / np.outer(std_diag, std_diag)

    poisson_var = mean_counts
    var_ratio = var_diag / poisson_var

    n = n_bins
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag_corr = corr_matrix[off_diag_mask]
    abs_off_diag = np.abs(off_diag_corr)

    median_abs_rho = np.median(abs_off_diag)
    mean_abs_rho = np.mean(abs_off_diag)
    max_abs_rho = np.max(abs_off_diag)
    frac_gt_01 = np.mean(abs_off_diag > 0.1)
    frac_gt_02 = np.mean(abs_off_diag > 0.2)
    frac_gt_05 = np.mean(abs_off_diag > 0.5)

    cond_number = np.linalg.cond(cov_matrix)
    idx = np.unravel_index(np.argmax(np.abs(corr_matrix - np.eye(n))), (n, n))

    # ─── 3. Report ───

    results_lines = []
    def log(s):
        print(s)
        results_lines.append(s)

    log("=" * 70)
    log(f"HMF COVARIANCE ANALYSIS ({mode_label}): 100 Quijote realisations")
    log("=" * 70)
    log("")
    bin_qualifier = ', trimmed data' if trimmed else ''
    log(f"Number of sigma bins used: {n_bins} (common to all 100 sims{bin_qualifier})")
    log(f"Sigma values: {np.array2string(sigma_ref, precision=4)}")
    log("")

    log("--- Diagonal: variance vs Poisson expectation ---")
    log(f"{'Bin':>3} {'sigma':>8} {'mean_N':>12} {'var(N)':>14} {'Poisson_var':>14} {'ratio':>8}")
    for j in range(n_bins):
        log(f"{j:3d} {sigma_ref[j]:8.4f} {mean_counts[j]:12.1f} {var_diag[j]:14.1f} {poisson_var[j]:14.1f} {var_ratio[j]:8.3f}")

    log("")
    log("--- Variance / Poisson ratio summary ---")
    log(f"  Min ratio:  {var_ratio.min():.4f}")
    log(f"  Max ratio:  {var_ratio.max():.4f}")
    log(f"  Mean ratio: {np.mean(var_ratio):.4f}")
    log(f"  Median ratio: {np.median(var_ratio):.4f}")
    log("")

    log("--- Off-diagonal correlation statistics ---")
    log(f"  Median |rho_ij| (i!=j): {median_abs_rho:.6f}")
    log(f"  Mean   |rho_ij| (i!=j): {mean_abs_rho:.6f}")
    log(f"  Max    |rho_ij| (i!=j): {max_abs_rho:.6f}")
    log(f"  Most correlated pair: bins {idx[0]} and {idx[1]} "
        f"(sigma={sigma_ref[idx[0]]:.4f}, {sigma_ref[idx[1]]:.4f}), rho={corr_matrix[idx]:.6f}")
    log(f"  Fraction |rho| > 0.1: {frac_gt_01:.4f}")
    log(f"  Fraction |rho| > 0.2: {frac_gt_02:.4f}")
    log(f"  Fraction |rho| > 0.5: {frac_gt_05:.4f}")
    log("")
    log(f"--- Condition number of covariance matrix: {cond_number:.2e} ---")
    log("")

    if median_abs_rho < 0.1:
        log("ASSESSMENT: The covariance matrix is approximately diagonal.")
        log(f"  (median |rho_ij| = {median_abs_rho:.6f} < 0.1)")
        approximately_diagonal = True
    else:
        log("ASSESSMENT: The covariance matrix has significant off-diagonal elements.")
        log(f"  (median |rho_ij| = {median_abs_rho:.6f} >= 0.1)")
        approximately_diagonal = False
    log("")

    # ─── 4. Correlation matrix heatmap ───

    os.makedirs('Final_Plots', exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    vmax = max(0.2, np.max(np.abs(off_diag_corr)))
    norm_cm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(corr_matrix, cmap='RdBu_r', norm=norm_cm, origin='upper')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r'$\rho_{ij}$', fontsize=14)

    tick_labels = [f"{s:.3f}" for s in sigma_ref]
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel(r'$\sigma$ bin', fontsize=12)
    ax.set_ylabel(r'$\sigma$ bin', fontsize=12)

    plt.tight_layout()
    plt.savefig(plot_outfile, dpi=150)
    plt.close()
    log(f"Saved correlation matrix heatmap to {plot_outfile}")
    log("")

    # ─── 5. If non-diagonal, compute impact on DL ───

    if not approximately_diagonal:
        log("=" * 70)
        log(f"COMPUTING IMPACT ON DL: Poisson vs Gaussian-with-covariance ({mode_label})")
        log("=" * 70)
        log("")

        ref_sim = 50
        ref_data = all_data[ref_sim][:n_bins]
        ref_sigma = ref_data[:, 0]
        ref_counts = ref_data[:, 1]
        ref_norm = ref_data[:, 3]

        sim50_entries = []
        with open(final_funcs_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(';')
                if len(parts) >= 5:
                    sim50_entries.append({
                        'source': parts[0],
                        'comp': int(parts[1]),
                        'nll': float(parts[2]),
                        'dl': float(parts[3]),
                        'func_with_params': parts[4],
                        'func_template': parts[5] if len(parts) > 5 else parts[4],
                    })

        log(f"Loaded {len(sim50_entries)} functions from {final_funcs_file}")
        log("")

        cov_inv = np.linalg.inv(cov_matrix)
        log_det_cov = np.linalg.slogdet(cov_matrix)[1]
        gauss_const = 0.5 * log_det_cov + 0.5 * n_bins * np.log(2 * np.pi)

        def poisson_nll(counts, ypred):
            if np.any(ypred <= 0):
                return np.inf
            return np.sum(ypred - counts * np.log(ypred))

        def gaussian_nll_with_cov(counts, ypred, cov_inv, gauss_const):
            residual = counts - ypred
            chi2 = residual @ cov_inv @ residual
            return 0.5 * chi2 + gauss_const

        def gaussian_nll_diag(counts, ypred, var_diag):
            residual = counts - ypred
            chi2 = np.sum(residual**2 / var_diag)
            log_det = np.sum(np.log(var_diag))
            return 0.5 * chi2 + 0.5 * log_det + 0.5 * n_bins * np.log(2 * np.pi)

        def eval_func_string(func_str, sigma):
            safe_dict = {
                'x': sigma, 'pow': np.power, 'Abs': np.abs,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'pi': np.pi, 'np': np,
            }
            try:
                result = eval(func_str, {"__builtins__": {}}, safe_dict)
                return np.atleast_1d(result).astype(float)
            except Exception:
                return None

        log(f"{'Source':>8} {'Comp':>4} {'Poisson_NLL':>14} {'Gauss_full':>14} {'Gauss_diag':>14} "
            f"{'chi2_full':>12} {'chi2_diag':>12} {'Function':>55}")
        log("-" * 140)

        results_for_ranking = []

        for entry in sim50_entries:
            func_str = entry['func_with_params']
            f_sigma = eval_func_string(func_str, ref_sigma)
            if f_sigma is None:
                log(f"{entry['source']:>8} {entry['comp']:4d} {'(eval err)':>14}")
                continue
            ypred = f_sigma * ref_norm
            if not np.all(np.isfinite(ypred)) or np.any(ypred <= 0):
                log(f"{entry['source']:>8} {entry['comp']:4d} {'(bad pred)':>14}")
                continue

            nll_p = poisson_nll(ref_counts, ypred)
            nll_gf = gaussian_nll_with_cov(ref_counts, ypred, cov_inv, gauss_const)
            nll_gd = gaussian_nll_diag(ref_counts, ypred, var_diag)
            residual = ref_counts - ypred
            chi2_full = residual @ cov_inv @ residual
            chi2_diag = np.sum(residual**2 / var_diag)

            log(f"{entry['source']:>8} {entry['comp']:4d} {nll_p:14.2f} {nll_gf:14.2f} {nll_gd:14.2f} "
                f"{chi2_full:12.2f} {chi2_diag:12.2f} {entry['func_template'][:55]:>55}")

            results_for_ranking.append({
                'source': entry['source'], 'comp': entry['comp'],
                'nll_poisson': nll_p, 'nll_gauss_full': nll_gf,
                'chi2_full': chi2_full, 'chi2_diag': chi2_diag,
                'func': entry['func_template'],
            })

        log("")
        log("--- Ranking comparison ---")
        log("(Lower chi2 = better fit; chi2_full accounts for bin correlations)")
        log("")

        by_poisson = sorted(results_for_ranking, key=lambda r: r['nll_poisson'])
        log("Ranking by Poisson NLL:")
        for i, r in enumerate(by_poisson):
            log(f"  {i+1:2d}. {r['source']:>8} comp={r['comp']:2d}  NLL_P={r['nll_poisson']:14.2f}  chi2_full={r['chi2_full']:12.2f}  {r['func'][:50]}")

        log("")
        by_chi2 = sorted(results_for_ranking, key=lambda r: r['chi2_full'])
        log("Ranking by chi2 (full covariance):")
        for i, r in enumerate(by_chi2):
            log(f"  {i+1:2d}. {r['source']:>8} comp={r['comp']:2d}  chi2_full={r['chi2_full']:12.2f}  NLL_P={r['nll_poisson']:14.2f}  {r['func'][:50]}")

        log("")
        log("--- Key question: does accounting for off-diagonal correlations change the ranking? ---")
        poisson_order = [r['func'] for r in by_poisson]
        chi2_order = [r['func'] for r in by_chi2]
        if poisson_order == chi2_order:
            log("  Rankings are IDENTICAL under Poisson and full-covariance Gaussian.")
        else:
            log("  Rankings DIFFER between Poisson and full-covariance Gaussian.")
            for i, (p, c) in enumerate(zip(poisson_order, chi2_order)):
                marker = " *" if p != c else ""
                log(f"    Position {i+1}: Poisson={by_poisson[i]['source']:>8} vs Gauss={by_chi2[i]['source']:>8}{marker}")

        log("")
        log("NOTE: The chi2 with full covariance penalises correlated residuals.")
        log("      For well-fitting functions (small residuals), off-diagonal terms")
        log("      contribute little. For poorly-fitting functions, correlated residuals")
        log("      can amplify or reduce chi2 depending on the sign of correlations.")
        log("      The Poisson likelihood remains appropriate if the ranking is unchanged,")
        log("      since DL differences (not absolute values) determine function selection.")
    else:
        log("")
        log("Since the covariance is approximately diagonal, the Poisson likelihood")
        log("(which assumes independent bins) is appropriate. No further analysis needed.")

    # ─── 6. Eigenvalue analysis ───
    log("")
    log("=" * 70)
    log("EIGENVALUE ANALYSIS OF COVARIANCE MATRIX")
    log("=" * 70)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    log(f"Eigenvalues (sorted): {np.array2string(eigenvalues, precision=2)}")
    log(f"All positive: {np.all(eigenvalues > 0)}")
    log(f"Condition number: {eigenvalues[-1]/eigenvalues[0]:.2e}")
    log("")

    # ─── 7. Full correlation matrix ───
    log("=" * 70)
    log("FULL CORRELATION MATRIX")
    log("=" * 70)
    for i in range(n_bins):
        row = " ".join(f"{corr_matrix[i,j]:7.4f}" for j in range(n_bins))
        log(f"  {row}")

    with open(results_outfile, 'w') as f:
        f.write('\n'.join(results_lines))

    print(f"\nResults saved to {results_outfile}")
    print(f"Correlation matrix heatmap saved to {plot_outfile}")


if __name__ == '__main__':
    main()
