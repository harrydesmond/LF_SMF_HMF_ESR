"""
propagated_impact.py
====================
Estimates the practical impact of using ESR functions instead of literature
functions for derived physical quantities:

  Part A: SMF — total stellar mass density rho_*
  Part B: LF  — total luminosity density rho_L
  Part C: HMF — predicted halo abundance n(>M) at several mass thresholds
  Part D: LF  — galaxy counts in an SDSS-like survey volume

Results are written to propagated_impact_results.txt.

Conventions
-----------
- SMF: x = M_* / 10^9 M_sun  (phi is per dex in M_*)
- LF : x = L   / 10^9 L_sun  (phi is per dex in L)
- HMF: x = sigma (rms mass fluctuation); fitted function = multiplicity f(sigma)

Physical densities
------------------
  rho_* [M_sun Mpc^-3] = 10^9 / ln(10) * integral{ phi(x) dx }
  rho_L [L_sun Mpc^-3] = 10^9 / ln(10) * integral{ phi(x) dx }

where the 10^9/ln(10) factor converts from (per dex) units.

HMF number density
------------------
  phi(log10 M) = f(sigma) * |d ln sigma / d log10 M|   [Mpc^-3 dex^-1]
  n(>M) = integral_{log10 M}^{inf} phi d(log10 M)

using the sigma-M relation and derivative from mass_variance_multiplier.txt.
"""

import numpy as np
import warnings
import os
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import gamma as scipy_gamma

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Function evaluator
# ---------------------------------------------------------------------------

def make_func(template):
    """Return a callable f(x) from a function-string template.

    Works for both filled-in strings (with numeric literals) and symbolic
    templates (with a0, a1, ...) — in the latter case the caller must
    supply params.
    """
    import re
    param_names = sorted(set(re.findall(r'a\d+', template)))

    def f(params_or_x, x=None):
        # Two calling conventions:
        #   f(x)               -- no free parameters (filled-in string)
        #   f(params, x)       -- symbolic template
        if x is None:
            x_arr = np.asarray(params_or_x, dtype=float)
            params = []
        else:
            x_arr = np.asarray(x, dtype=float)
            params = list(params_or_x)

        ns = {
            'x'   : x_arr,
            'pow' : lambda a, b: np.float_power(np.abs(a), b),
            'Abs' : np.abs,
            'abs' : np.abs,
            'exp' : np.exp,
            'log' : lambda a: np.log(np.abs(a)),
            'sqrt': np.sqrt,
            'pi'  : np.pi,
            'Gamma': scipy_gamma,
            'gamma': scipy_gamma,
        }
        for name, val in zip(param_names, params):
            ns[name] = val
        with np.errstate(all='ignore'):
            result = eval(template, {"__builtins__": {}}, ns)
        return np.asarray(result, dtype=float)

    return f


def eval_str(fcn_str, x_vals):
    """Evaluate a filled-in function string (no free parameters)."""
    x = np.asarray(x_vals, dtype=float)
    ns = {
        'x'   : x,
        'pow' : lambda a, b: np.float_power(np.abs(a), b),
        'Abs' : np.abs,
        'abs' : np.abs,
        'exp' : np.exp,
        'log' : lambda a: np.log(np.abs(a)),
        'log10': lambda a: np.log10(np.abs(a)),
        'sqrt': np.sqrt,
        'pi'  : np.pi,
        'Gamma': scipy_gamma,
        'gamma': scipy_gamma,
    }
    with np.errstate(all='ignore'):
        result = eval(fcn_str, {"__builtins__": {}}, ns)
    return np.asarray(result, dtype=float)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_final_functions(filepath):
    """Load a *_final_functions.txt file.

    Returns list of dicts with keys: source, comp, DL, NLL, plot_fcn, blank_fcn
    """
    records = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';')
            if len(parts) < 5:
                continue
            rec = {
                'source'  : parts[0].strip(),
                'comp'    : int(parts[1].strip()),
                'DL'      : float(parts[2].strip()),
                'NLL'     : float(parts[3].strip()),
                'plot_fcn': parts[4].strip(),
                'blank_fcn': parts[5].strip() if len(parts) > 5 else '',
            }
            records.append(rec)
    return records


def best_esr(records, prefer=('ESR_C', 'ESR_T', 'ESR')):
    """Return the best (lowest DL) ESR function record, preferring ESR_C."""
    for src_pref in prefer:
        candidates = [r for r in records if r['source'] == src_pref]
        if candidates:
            return min(candidates, key=lambda r: r['DL'])
    # Fallback: any ESR-prefixed
    candidates = [r for r in records if r['source'].startswith('ESR')]
    if candidates:
        return min(candidates, key=lambda r: r['DL'])
    return None


def get_record(records, source):
    """Return the record with matching source (first match, lowest DL)."""
    candidates = [r for r in records if r['source'] == source]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r['DL'])


# ---------------------------------------------------------------------------
# Part A & B: stellar mass / luminosity density
# ---------------------------------------------------------------------------

def density_integral(fcn_str, x_min, x_max, n_pts=200_000):
    """Compute int_{x_min}^{x_max} phi(x) dx on a log-uniform grid.

    phi is given per dex (d log10 x), so phi(x) = y(x) / (x * ln10).
    The stellar-mass density is rho = 10^9 * int phi(x) dx in appropriate units.
    BUT: the function stored in final_functions.txt IS phi (per unit x, not per dex),
    because the fitting uses a Gaussian likelihood on the log10(phi) values and
    the x-axis has already absorbed the 1/ln10 Jacobian.

    Actually, check: the physicality_checks.py computes:
        rho = 1e9 / ln(10) * integral(f(x) dx)
    where f(x) is the function value (phi in per-dex units, i.e. dn/d(log10 x)).
    So the stored function IS phi = dn/d(log10 x), and:
        rho = 10^9 * integral{ phi(x) / (x ln 10) } dx ... no

    From physicality_checks.py: rho = 1e9 / ln(10) * int f dx
    so the stored f = phi(x) is per unit x (linear), and
        rho [M_sun Mpc^-3] = 10^9 / ln(10) * int phi(x) dx
    This factor converts: phi is in units of (10^9 M_sun)^-1 Mpc^-3, and
    10^9 converts to M_sun Mpc^-3, /ln(10) converts from per-dex to per-linear.

    For the MASS-WEIGHTED density rho_* = int M_* phi dM_*:
        = 10^9 * int x * phi(x) * 10^9 dx / 10^9
        = 10^9 * int x * phi(x) dx   [in M_sun Mpc^-3]
    because x = M_*/10^9, so M_* = 10^9 x.

    Actually: phi [Mpc^-3 dex^-1], M_* = 10^9 x [M_sun], dM_* = 10^9 dx
        rho_* = int M_* * phi(M_*) dM_*
              = int (10^9 x) * phi(x) * (10^9 dx)   ... where phi(x) is per unit x?

    Let me reconcile with physicality_checks.py which gives rho = 1e9/ln10 * int f dx.
    This is the NUMBER density (zeroth moment); for mass/lum density (first moment):
        rho_* = int M_* phi dM_* = int (10^9 x) * phi(x)/ln(10) * (1/x) * 10^9 dx
             ... if phi is per dex then dN/dlogM = phi, dN = phi * d(logM) = phi/(x ln10) dx
             so dN/dx = phi/(x ln10)
             rho_* = int M_* * (dN/dx) dx = int (10^9 x) * phi/(x ln10) * 10^9 dx
                   = 10^18 / ln10 * int phi dx

    But physicality_checks uses rho = 1e9/ln10 * int f dx for the ZEROTH moment (number density).
    So for the FIRST moment (mass density):
        rho_* = int M_* * dN/dlogM * d(logM)
              = int (10^9 x) * phi(x) * d(log10 x)
              = int (10^9 x) * phi(x) * (1/(x ln10)) dx
              = 10^9/ln10 * int phi(x) dx

    Wait, that's the same formula as the number density! That can't be right.

    Let me be explicit:
        phi(x) = dN/d(log10 x)   [Mpc^-3 dex^-1], where x = M*/10^9 Msun
        dN = phi * d(log10 x) = phi/(x ln10) * dx
        rho_* = int M_* dN = int (10^9 x) * phi/(x ln10) dx
              = 10^9/ln10 * int phi dx   <-- THIS is correct for rho_*

    And number density:
        n = int dN = int phi/(x ln10) dx   ≠ 10^9/ln10 * int phi dx

    So physicality_checks.py is computing rho_* (NOT n_total)?! Let me reread...
    "rho = 1e9 / ln(10) * integral(f(x) dx)" -- yes this is rho_* (mass density)!
    The 10^9 factor comes from x = M*/10^9 so M_* = 10^9 * x.

    But WAIT: the first moment for the mass density should be:
        rho_* = int M_* * phi(M_*) dM_*
    not int phi dM.

    Let me redo:
        rho_* = int M_* * (dN/dM_*) dM_*
        dN/dM_* = phi / (M_* ln10)    [since phi = dN/d(log10 M_*)]
        rho_* = int M_* * phi / (M_* ln10) dM_* = int phi / ln10 dM_*
              = int phi / ln10 * 10^9 dx    [M_* = 10^9 x, dM_* = 10^9 dx]
              = 10^9/ln10 * int phi(x) dx   <-- confirmed!

    So rho_* = 10^9/ln10 * int phi(x) dx   in M_sun Mpc^-3.

    But we want rho_* = int M_* phi dM_* which is THE MASS-WEIGHTED DENSITY (cosmic stellar
    mass density). This is the standard definition! Let me verify again:

    rho_* = int_{M_min}^{M_max} M_* * (dn/dM_*) dM_*
          = int phi/ln10 * dM_*          [since dn/dM_* = phi/(M_* ln10), so M_* * dn/dM_* = phi/ln10]
          = 10^9/ln10 * int phi dx       YES! Confirmed.

    So the physicality_checks formula IS for rho_* (cosmic stellar mass density).
    Good. Now for the LUMINOSITY density (Part B):
        rho_L = int L * phi(L) dL = 10^9/ln10 * int phi(x) dx   [L_sun Mpc^-3]
    Same formula.
    """
    x = np.logspace(np.log10(x_min), np.log10(x_max), n_pts)
    y = eval_str(fcn_str, x)
    y_clean = np.where(np.isfinite(y) & (y > 0), y, 0.0)
    integral = np.trapz(y_clean, x)
    rho = 1e9 / np.log(10) * integral
    return rho, integral


def compute_smf_density(records, dataset_name, x_min_data, x_max_data,
                         x_min_wide=0.01, x_max_wide=1000.0, lines=None):
    """Compute rho_* for ESR, Schechter, and Bernardi from a final_functions file."""
    if lines is None:
        lines = []

    esr_rec  = best_esr(records)
    sch_rec  = get_record(records, 'Sch.')
    ber_rec  = get_record(records, 'Ber.')
    sau_rec  = get_record(records, 'Sau.')

    results = {}

    for label, rec in [('ESR', esr_rec), ('Schechter', sch_rec), ('Bernardi', ber_rec),
                       ('Saunders', sau_rec)]:
        if rec is None:
            results[label] = (None, None)
            continue
        fcn = rec['plot_fcn']
        rho_data, _ = density_integral(fcn, x_min_data, x_max_data)
        rho_wide, _ = density_integral(fcn, x_min_wide, x_max_wide)
        results[label] = (rho_data, rho_wide)
        src_info = f"{rec['source']} comp={rec['comp']}"
        lines.append(f"  {label:12s} ({src_info}): "
                     f"rho_*(data range) = {rho_data:.4e} M_sun Mpc^-3 | "
                     f"rho_*(wide [0.01,1000]) = {rho_wide:.4e} M_sun Mpc^-3")
        lines.append(f"    fcn: {fcn[:80]}{'...' if len(fcn)>80 else ''}")

    # Fractional differences vs Schechter
    for label in ('ESR', 'Bernardi', 'Saunders'):
        rho_data_ref, rho_wide_ref = results.get('Schechter', (None, None))
        rho_data_x,   rho_wide_x   = results.get(label, (None, None))
        if rho_data_ref and rho_data_x:
            frac_data = (rho_data_x - rho_data_ref) / rho_data_ref
            frac_wide = (rho_wide_x - rho_wide_ref) / rho_wide_ref
            lines.append(f"  Delta rho_* ({label} vs Schechter): "
                         f"data range = {frac_data:+.3f} ({frac_data*100:+.1f}%) | "
                         f"wide range = {frac_wide:+.3f} ({frac_wide*100:+.1f}%)")

    # ESR relative to Saunders (parallels the ESR-vs-Schechter quote in the text)
    rho_data_sau, rho_wide_sau = results.get('Saunders', (None, None))
    rho_data_esr, rho_wide_esr = results.get('ESR', (None, None))
    if rho_data_sau and rho_data_esr:
        fd = (rho_data_esr - rho_data_sau) / rho_data_sau
        fw = (rho_wide_esr - rho_wide_sau) / rho_wide_sau
        lines.append(f"  Delta rho_* (ESR vs Saunders): "
                     f"data range = {fd*100:+.1f}% | wide range = {fw*100:+.1f}%")

    return results, lines


# ---------------------------------------------------------------------------
# Part C: HMF halo abundance
# ---------------------------------------------------------------------------

def compute_hmf_abundance(records, mv_file, thresholds_logM, lines=None):
    """Compute n(>M) for each function in records at each threshold.

    Parameters
    ----------
    records : list of dicts from load_final_functions
    mv_file : path to mass_variance_multiplier.txt (logM, sigma, d_ln_sigma/d_log10_M)
    thresholds_logM : list of log10(M/[Msun/h]) values
    """
    if lines is None:
        lines = []

    mv = np.loadtxt(mv_file)
    logM_mv = mv[:, 0]       # log10(M / [Msun/h])
    sigma_mv = mv[:, 1]      # rms mass fluctuation
    factor_mv = mv[:, 2]     # |d ln sigma / d log10 M|
    delta_logM = 0.2         # bin width in log10 M

    # Labels we want to compare
    wanted = {
        'ESR_best': best_esr(records),
        'P.Sch.'  : get_record(records, 'P.Sch.'),
        'Warren'  : get_record(records, 'War.'),
        'Tinker'  : get_record(records, 'Tin.'),
        'Sheth-Tormen': get_record(records, 'S-T.'),
        'Jenkins' : get_record(records, 'Jen.'),
    }

    abundances = {}  # label -> array of n(>M) for each threshold

    for label, rec in wanted.items():
        if rec is None:
            lines.append(f"  {label}: not found in records")
            continue
        fcn = rec['plot_fcn']
        # Evaluate f(sigma) at all sigma values in mass_variance_multiplier
        f_sigma = eval_str(fcn, sigma_mv)
        f_sigma = np.where(np.isfinite(f_sigma), f_sigma, 0.0)

        # dn/d(log10 M) = f(sigma) * |d ln sigma / d log10 M|
        phi = f_sigma * factor_mv

        n_above = []
        for logM_thresh in thresholds_logM:
            mask = logM_mv >= logM_thresh
            n = np.sum(phi[mask] * delta_logM)  # integrate
            n_above.append(n)
        abundances[label] = np.array(n_above)

        src_info = f"{rec['source']} comp={rec['comp']}"
        line_parts = [f"  {label:12s} ({src_info}):"]
        for logM_thresh, n in zip(thresholds_logM, n_above):
            line_parts.append(f"    n(>10^{logM_thresh:.0f} Msun/h) = {n:.4e} Mpc^-3")
        lines.extend(line_parts)

    # Fractional differences vs ESR_best
    esr_n = abundances.get('ESR_best')
    if esr_n is not None:
        for label in ('P.Sch.', 'Warren', 'Tinker', 'Sheth-Tormen', 'Jenkins'):
            n_lit = abundances.get(label)
            if n_lit is None:
                continue
            lines.append(f"  Fractional diff (ESR - {label}) / {label}:")
            for logM_thresh, n_esr, n_l in zip(thresholds_logM, esr_n, n_lit):
                if n_l > 0:
                    frac = (n_esr - n_l) / n_l
                    lines.append(f"    M > 10^{logM_thresh:.0f} Msun/h: "
                                 f"delta = {frac:+.3f} ({frac*100:+.1f}%)")
                else:
                    lines.append(f"    M > 10^{logM_thresh:.0f} Msun/h: n_lit=0, cannot compute")

    return abundances, lines


# ---------------------------------------------------------------------------
# Part D: Galaxy counts in survey volume
# ---------------------------------------------------------------------------

def galaxy_count_survey(records, dataset_name, x_min_data, x_max_data,
                          L_threshold_x, survey_volume_Mpc3, lines=None):
    """Estimate total galaxy count above a luminosity threshold in a survey volume."""
    if lines is None:
        lines = []

    esr_rec = best_esr(records)
    sch_rec = get_record(records, 'Sch.')
    sau_rec = get_record(records, 'Sau.')

    results = {}
    for label, rec in [('ESR', esr_rec), ('Schechter', sch_rec), ('Saunders', sau_rec)]:
        if rec is None:
            results[label] = None
            continue
        fcn = rec['plot_fcn']
        # Integrate phi from L_threshold to data_max and then wide max
        x_int_max = max(x_max_data, 1000.0)
        rho_above, _ = density_integral(fcn, L_threshold_x, x_int_max)
        # But we want NUMBER density not rho_L, so:
        #   n_above = integral_{x_thresh}^{inf} phi(x) / (x ln10) dx
        x = np.logspace(np.log10(L_threshold_x), np.log10(x_int_max), 100_000)
        y = eval_str(fcn, x)
        y_clean = np.where(np.isfinite(y) & (y > 0), y, 0.0)
        # dN/dx = phi/(x ln10)
        integrand = y_clean / (x * np.log(10))
        n_density = np.trapz(integrand, x)   # Mpc^-3
        N_survey = n_density * survey_volume_Mpc3
        results[label] = (n_density, N_survey)
        src_info = f"{rec['source']} comp={rec['comp']}"
        lines.append(f"  {label:12s} ({src_info}): "
                     f"n(>L_thresh) = {n_density:.4e} Mpc^-3 | "
                     f"N_survey = {N_survey:.3e}")

    esr_n, _ = results.get('ESR', (None, None)) or (None, None)
    sch_n, _ = results.get('Schechter', (None, None)) or (None, None)
    sau_n, _ = results.get('Saunders', (None, None)) or (None, None)
    if esr_n and sch_n:
        frac = (esr_n - sch_n) / sch_n
        lines.append(f"  Fractional diff (ESR - Schechter) / Schechter: "
                     f"{frac:+.3f} ({frac*100:+.1f}%)")
    if esr_n and sau_n:
        frac = (esr_n - sau_n) / sau_n
        lines.append(f"  Fractional diff (ESR - Saunders) / Saunders: "
                     f"{frac:+.3f} ({frac*100:+.1f}%)")

    return results, lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_lines = []

    def hdr(title):
        output_lines.append("")
        output_lines.append("=" * 70)
        output_lines.append(f"  {title}")
        output_lines.append("=" * 70)

    output_lines.append("PROPAGATED IMPACT OF ESR vs LITERATURE FUNCTIONS")
    output_lines.append("=" * 70)
    output_lines.append("")

    # -----------------------------------------------------------------------
    # Part A: SMF — total stellar mass density
    # -----------------------------------------------------------------------
    hdr("PART A: SMF — Total Stellar Mass Density  rho_*")
    output_lines.append("")
    output_lines.append("Convention: x = M_* / 10^9 M_sun; phi = dn/d(log10 M_*)")
    output_lines.append("  rho_* = 10^9 / ln(10) * integral phi(x) dx  [M_sun Mpc^-3]")
    output_lines.append("")

    for dataset, fname, x_min, x_max in [
        ('SMF Sersic',
         os.path.join(BASE, 'SMF_Ser_M_final_functions.txt'),
         1.12, 2238.7),
        ('SMF cmodel',
         os.path.join(BASE, 'SMF_cmodel_M_final_functions.txt'),
         1.12, 1412.5),
    ]:
        output_lines.append(f"--- {dataset} (data x range: [{x_min:.2f}, {x_max:.1f}]) ---")
        recs = load_final_functions(fname)
        _, output_lines = compute_smf_density(
            recs, dataset, x_min, x_max,
            x_min_wide=0.01, x_max_wide=1000.0,
            lines=output_lines)
        output_lines.append("")

    # -----------------------------------------------------------------------
    # Part B: LF — total luminosity density
    # -----------------------------------------------------------------------
    hdr("PART B: LF — Total Luminosity Density  rho_L")
    output_lines.append("")
    output_lines.append("Convention: x = L / 10^9 L_sun; phi = dn/d(log10 L)")
    output_lines.append("  rho_L = 10^9 / ln(10) * integral phi(x) dx  [L_sun Mpc^-3]")
    output_lines.append("")

    for dataset, fname, x_min, x_max in [
        ('LF Sersic',
         os.path.join(BASE, 'LF_Ser_L_final_functions.txt'),
         0.946, 597.0),
        ('LF cmodel',
         os.path.join(BASE, 'LF_cmodel_L_final_functions.txt'),
         0.887, 465.6),
    ]:
        output_lines.append(f"--- {dataset} (data x range: [{x_min:.3f}, {x_max:.1f}]) ---")
        recs = load_final_functions(fname)
        _, output_lines = compute_smf_density(
            recs, dataset, x_min, x_max,
            x_min_wide=0.01, x_max_wide=1000.0,
            lines=output_lines)
        output_lines.append("")

    # -----------------------------------------------------------------------
    # Part C: HMF — predicted halo abundance
    # -----------------------------------------------------------------------
    hdr("PART C: HMF — Predicted Halo Abundance n(>M)")
    output_lines.append("")
    output_lines.append("Convention: x = sigma (rms mass fluctuation)")
    output_lines.append("  f(sigma) = multiplicity function (fitted directly)")
    output_lines.append("  phi = f(sigma) * |d ln sigma / d log10 M|   [Mpc^-3 dex^-1]")
    output_lines.append("  n(>M) = sum_{log10 M' > log10 M} phi * delta_logM  [Mpc^-3]")
    output_lines.append("  (discrete sum over bins in mass_variance_multiplier.txt)")
    output_lines.append("  Masses in units of M_sun/h")
    output_lines.append("")

    hmf_recs = load_final_functions(os.path.join(BASE, 'hmf_50_final_functions_fiducial.txt'))
    mv_file  = os.path.join(BASE, 'mass_variance_multiplier.txt')
    thresholds_logM = [13.0, 14.0, 15.0]   # log10(M / [Msun/h])

    _, output_lines = compute_hmf_abundance(
        hmf_recs, mv_file, thresholds_logM, lines=output_lines)
    output_lines.append("")

    # -----------------------------------------------------------------------
    # Part D: galaxy counts in SDSS-like survey
    # -----------------------------------------------------------------------
    hdr("PART D: Galaxy Counts in SDSS-like Survey Volume")
    output_lines.append("")
    output_lines.append("Survey volume: ~10^8 Mpc^3 (bright-end SDSS)")
    output_lines.append("Luminosity threshold: L > 10^10 L_sun  =>  x > 10.0")
    output_lines.append("  N_survey = n(>L_thresh) * V_survey")
    output_lines.append("")

    survey_volume = 1e8   # Mpc^3
    L_thresh_x   = 10.0  # x = L / 10^9 Lsun = 10 => L = 10^10 Lsun

    for dataset, fname, x_min, x_max in [
        ('LF Sersic',
         os.path.join(BASE, 'LF_Ser_L_final_functions.txt'),
         0.946, 597.0),
        ('LF cmodel',
         os.path.join(BASE, 'LF_cmodel_L_final_functions.txt'),
         0.887, 465.6),
    ]:
        output_lines.append(f"--- {dataset} ---")
        recs = load_final_functions(fname)
        _, output_lines = galaxy_count_survey(
            recs, dataset, x_min, x_max,
            L_threshold_x=L_thresh_x,
            survey_volume_Mpc3=survey_volume,
            lines=output_lines)
        output_lines.append("")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    hdr("SUMMARY")
    output_lines.append("")
    output_lines.append(
        "Fractional differences (ESR - literature) / literature for key quantities.")
    output_lines.append(
        "See individual sections above for values at each dataset/threshold.")
    output_lines.append("")

    # Print to stdout and save
    report = "\n".join(output_lines)
    print(report)

    out_path = os.path.join(BASE, 'propagated_impact_results.txt')
    with open(out_path, 'w') as fh:
        fh.write(report + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
